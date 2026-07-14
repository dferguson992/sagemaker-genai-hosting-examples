#!/usr/bin/env python3
"""
Download benchmark datasets from HuggingFace and upload to S3 in JSONL format.

Converts datasets into the format expected by SageMaker AI Benchmark (AIPerf):
- JSONL with OpenAI chat completion format
- Each line is a JSON object with "messages" array

S3 path preserves the HuggingFace dataset ID:
  s3://<bucket>/datasets/launch/gov_report/data.jsonl

Usage:
    python download_datasets.py --dataset=launch/gov_report --region=us-west-2
    python download_datasets.py --dataset=launch/gov_report --region=us-west-2 --split=test --max-samples=500
    python download_datasets.py --submit --dataset=launch/gov_report --region=us-west-2
    python download_datasets.py --validate

Output: s3://sagemaker-benchmark-{region}-{account}/datasets/{dataset_id}/data.jsonl
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import boto3

ENTRYPOINT_MODE = os.environ.get("SM_PROCESSING_MODE") == "download_datasets"

# Dataset registry: maps HuggingFace IDs to conversion configs
DATASETS = {
    "launch/gov_report": {
        "hf_config": "plain_text",
        "split": "test",
        "max_samples": 500,
        "description": "Government reports — long-context summarization (13K+ tokens input)",
        "prompt_template": "Summarize the following government report in detail:\n\n{document}",
        "document_field": "document",
        "summary_field": "summary",
    },
}


def get_account(region):
    return boto3.client("sts", region_name=region).get_caller_identity()["Account"]


def ensure_bucket(bucket, region):
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        try:
            if region == "us-east-1":
                s3.create_bucket(Bucket=bucket)
            else:
                s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
            print(f"  ✓ Created bucket: {bucket}")
        except Exception:
            pass


def check_s3_exists(bucket, key, region):
    s3 = boto3.client("s3", region_name=region)
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
        # Consider empty files as non-existent (failed prior upload)
        return resp.get("ContentLength", 0) > 0
    except Exception:
        return False


def dataset_id_to_s3_prefix(dataset_id):
    """Preserve the HuggingFace dataset ID as-is in the S3 path.
    e.g. 'launch/gov_report' -> 'datasets/launch/gov_report/'
    """
    return f"datasets/{dataset_id}/"


def download_and_convert(dataset_id, config, max_samples=None, split=None):
    """Download from HuggingFace and convert to JSONL chat format.

    Uses the auto-converted Parquet files on refs/convert/parquet branch
    (the official approach for script-based datasets since datasets>=3.0).
    Falls back to the Dataset Viewer REST API if parquet loading fails.
    """
    split = split or config["split"]
    max_samples = max_samples or config["max_samples"]

    print(f"  ⬇️  Loading {dataset_id} (split={split})...")

    # Primary: load from the auto-converted parquet branch
    try:
        rows = _load_via_parquet(dataset_id, config, split, max_samples)
    except Exception as e:
        print(f"  ⚠️  Parquet load failed: {e}")
        print(f"  → Falling back to Dataset Viewer REST API...")
        rows = _load_via_api(dataset_id, split, max_samples)

    records = []
    for i, row in enumerate(rows):
        if i >= max_samples:
            break
        document = row.get(config["document_field"], "")
        if not document or len(document) < 100:
            continue

        # Format as OpenAI chat completion messages
        prompt = config["prompt_template"].format(document=document)
        record = {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        records.append(record)

    print(f"  ✓ Converted {len(records)} samples to chat format")
    return records


def _load_via_parquet(dataset_id, config, split, max_samples):
    """Load from the HuggingFace auto-converted parquet branch.

    HuggingFace converts all datasets (including script-based) to Parquet on
    the refs/convert/parquet branch. This is the official replacement for
    trust_remote_code / loading scripts.

    Uses: datasets-server API to discover parquet URLs, then loads via pandas.
    """
    import urllib.request

    # 1. Get parquet file URLs from the datasets-server API
    api_url = f"https://datasets-server.huggingface.co/parquet?dataset={dataset_id}"
    print(f"    Querying parquet index: {api_url}")
    with urllib.request.urlopen(api_url) as resp:
        parquet_info = json.loads(resp.read().decode())

    # 2. Find parquet files for our config + split
    hf_config = config.get("hf_config", "plain_text")
    parquet_urls = [
        f["url"] for f in parquet_info["parquet_files"]
        if f["config"] == hf_config and f["split"] == split
    ]
    if not parquet_urls:
        raise ValueError(f"No parquet files found for config={hf_config}, split={split}")

    print(f"    Found {len(parquet_urls)} parquet file(s) for {hf_config}/{split}")

    # 3. Load parquet files into rows
    import pandas as pd

    dfs = []
    for url in parquet_urls:
        print(f"    Downloading: {url.split('/')[-1]}")
        df = pd.read_parquet(url)
        dfs.append(df)
        # Stop early if we already have enough rows
        total_rows = sum(len(d) for d in dfs)
        if total_rows >= max_samples:
            break

    combined = pd.concat(dfs, ignore_index=True)
    rows = combined.head(max_samples).to_dict("records")
    print(f"  ✓ Loaded {len(rows)} rows from parquet")
    return rows


def _load_via_api(dataset_id, split, max_samples):
    """Fallback: download rows via HuggingFace Dataset Viewer REST API."""
    import urllib.request

    rows = []
    offset = 0
    page_size = 100

    while len(rows) < max_samples:
        url = (
            f"https://datasets-server.huggingface.co/rows"
            f"?dataset={dataset_id}&config=plain_text&split={split}"
            f"&offset={offset}&length={min(page_size, max_samples - len(rows))}"
        )
        try:
            with urllib.request.urlopen(url) as resp:
                data = json.loads(resp.read().decode())
            page_rows = [r["row"] for r in data.get("rows", [])]
            if not page_rows:
                break
            rows.extend(page_rows)
            offset += len(page_rows)
            print(f"    fetched {len(rows)}/{max_samples} rows...")
        except Exception as e:
            print(f"  ✗ API request failed: {e}")
            break

    print(f"  ✓ Loaded {len(rows)} rows via API")
    return rows


def upload_dataset(records, dataset_id, region, account):
    """Write JSONL and upload to S3."""
    bucket = f"sagemaker-benchmark-{region}-{account}"
    prefix = dataset_id_to_s3_prefix(dataset_id)
    s3_key = f"{prefix}data.jsonl"
    s3_uri = f"s3://{bucket}/{prefix}"

    # Check if already exists
    if check_s3_exists(bucket, s3_key, region):
        print(f"  ⏭️  Already exists: {s3_uri}data.jsonl")
        return s3_uri

    ensure_bucket(bucket, region)

    # Write JSONL to temp file then upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
        tmp_path = f.name

    s3 = boto3.client("s3", region_name=region)
    print(f"  ⬆️  Uploading to s3://{bucket}/{s3_key} ({len(records)} records)...")
    s3.upload_file(tmp_path, bucket, s3_key)
    os.unlink(tmp_path)

    print(f"  ✓ Uploaded: {s3_uri}")
    return s3_uri


def submit_job(args):
    """Submit as a SageMaker Processing Job (no local download needed)."""
    region = args.region
    account = get_account(region)
    bucket = f"sagemaker-benchmark-{region}-{account}"

    # Need a role — try to find from a benchmarks config or env
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if not role:
        # Try loading from a config file if available
        try:
            import yaml
            for cfg_path in ["../benchmarks-phase2.yaml", "../benchmarks.yaml", "../benchmarks-nemotron-ultra.yaml"]:
                if os.path.exists(cfg_path):
                    with open(cfg_path) as f:
                        cfg = yaml.safe_load(f)
                    role = cfg.get("sagemaker_defaults", {}).get("role_arn")
                    if role:
                        break
        except Exception:
            pass
    if not role:
        print("ERROR: No role ARN found. Set SAGEMAKER_ROLE_ARN or ensure a benchmarks YAML with role_arn exists.")
        sys.exit(1)

    job_name = f"dl-dataset-{datetime.now().strftime('%m%d-%H%M')}"[:63]
    sm = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name=region)

    # Upload this script to S3
    ensure_bucket(bucket, region)
    prefix = f"processing-configs/{job_name}"
    script_path = os.path.abspath(__file__)
    s3.put_object(Bucket=bucket, Key=f"{prefix}/download_datasets.py", Body=open(script_path).read())

    # Build container args
    container_args = [
        "--dataset", args.dataset,
        "--region", region,
    ]
    if args.split:
        container_args += ["--split", args.split]
    if args.max_samples:
        container_args += ["--max-samples", str(args.max_samples)]
    if args.document_field:
        container_args += ["--document-field", args.document_field]
    if args.prompt_template:
        container_args += ["--prompt-template", args.prompt_template]
    if args.hf_config:
        container_args += ["--hf-config", args.hf_config]

    print(f"\n{'='*60}")
    print(f"Submitting Processing Job: {job_name}")
    print(f"  Region: {region}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Instance: ml.m5.xlarge")
    print(f"{'='*60}\n")

    sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m5.xlarge",
                "VolumeSizeInGB": 100,
            }
        },
        AppSpecification={
            "ImageUri": f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.5.1-cpu-py311",
            "ContainerEntrypoint": [
                "bash", "-c",
                "pip install -q pandas pyarrow && "
                f"SM_PROCESSING_MODE=download_datasets python3 /opt/ml/processing/input/script/download_datasets.py "
                + " ".join(container_args),
            ],
        },
        ProcessingInputs=[
            {
                "InputName": "script",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{prefix}/",
                    "LocalPath": "/opt/ml/processing/input/script",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],
        ProcessingOutputConfig={
            "Outputs": [{
                "OutputName": "logs",
                "S3Output": {
                    "S3Uri": f"s3://{bucket}/processing-results/{job_name}/",
                    "LocalPath": "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                },
            }]
        },
        RoleArn=role,
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        NetworkConfig={"EnableNetworkIsolation": False},
    )

    print(f"  ✓ Job submitted: {job_name}")
    print(f"  Monitor: aws sagemaker describe-processing-job --processing-job-name {job_name} --region {region}")
    print(f"\n  Output will be at: s3://{bucket}/{dataset_id_to_s3_prefix(args.dataset)}data.jsonl")


def validate():
    """Show available datasets and their configs."""
    print("Available datasets:\n")
    for dataset_id, config in DATASETS.items():
        print(f"  {dataset_id}")
        print(f"    split: {config['split']}")
        print(f"    max_samples: {config['max_samples']}")
        print(f"    description: {config['description']}")
        print(f"    s3 path: datasets/{dataset_id}/data.jsonl")
        print()


def run_download(dataset_id, region, max_samples=None, split=None, document_field=None, prompt_template=None, hf_config=None):
    """Execute the download (locally or inside Processing Job)."""
    if dataset_id in DATASETS:
        config = dict(DATASETS[dataset_id])  # copy so we can override
    else:
        # Ad-hoc dataset — require document_field at minimum
        if not document_field:
            print(f"  ✗ Unknown dataset: {dataset_id}")
            print(f"  For datasets not in the registry, provide --document-field")
            print(f"  Available registered datasets: {', '.join(DATASETS.keys())}")
            sys.exit(1)
        config = {
            "hf_config": hf_config or "default",
            "split": split or "test",
            "max_samples": max_samples or 500,
            "description": f"Custom dataset: {dataset_id}",
            "prompt_template": prompt_template or "{document}",
            "document_field": document_field,
        }

    # CLI overrides take precedence over registry
    if document_field:
        config["document_field"] = document_field
    if prompt_template:
        config["prompt_template"] = prompt_template
    if hf_config:
        config["hf_config"] = hf_config

    account = get_account(region)

    print(f"\n{'='*60}")
    print(f"[download] {dataset_id}")
    print(f"  source: huggingface.co/datasets/{dataset_id}")
    print(f"  document_field: {config['document_field']}")
    print(f"  target: s3://sagemaker-benchmark-{region}-{account}/{dataset_id_to_s3_prefix(dataset_id)}")
    print(f"{'='*60}")

    records = download_and_convert(dataset_id, config, max_samples, split)
    s3_uri = upload_dataset(records, dataset_id, region, account)

    print(f"\n{'─'*60}")
    print(f"Done. Use in workload config:")
    print(f"  dataset: {s3_uri}")
    print(f"{'─'*60}")


def main():
    if ENTRYPOINT_MODE:
        # Running inside Processing Job — install deps already handled by entrypoint
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", required=True)
        parser.add_argument("--region", required=True)
        parser.add_argument("--split", default=None)
        parser.add_argument("--max-samples", type=int, default=None)
        parser.add_argument("--document-field", default=None, help="Column name containing the document text")
        parser.add_argument("--prompt-template", default=None, help="Prompt template with {document} placeholder")
        parser.add_argument("--hf-config", default=None, help="HuggingFace dataset config/subset name")
        args = parser.parse_args()
        run_download(args.dataset, args.region, args.max_samples, args.split,
                     args.document_field, args.prompt_template, args.hf_config)
        return

    parser = argparse.ArgumentParser(
        description="Download benchmark datasets from HuggingFace → S3 (JSONL chat format).",
        epilog="""
examples:
  %(prog)s --dataset=launch/gov_report --region=us-west-2
  %(prog)s --dataset=launch/gov_report --region=us-west-2 --max-samples=100
  %(prog)s --submit --dataset=launch/gov_report --region=us-west-2
  %(prog)s --dataset=my-org/my-data --document-field=text --region=us-west-2
  %(prog)s --validate
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", help="HuggingFace dataset ID (e.g. launch/gov_report)")
    parser.add_argument("--region", default="us-west-2", help="AWS region for S3 bucket")
    parser.add_argument("--split", help="Dataset split (default: from registry or 'test')")
    parser.add_argument("--max-samples", type=int, help="Max samples to include (default: from registry or 500)")
    parser.add_argument("--document-field", default=None,
                        help="Column name containing the document text (overrides registry)")
    parser.add_argument("--prompt-template", default=None,
                        help="Prompt template with {document} placeholder (overrides registry)")
    parser.add_argument("--hf-config", default=None,
                        help="HuggingFace dataset config/subset name (overrides registry)")
    parser.add_argument("--submit", action="store_true", help="Submit as SageMaker Processing Job (no local download)")
    parser.add_argument("--validate", action="store_true", help="Show available datasets and exit")
    args = parser.parse_args()

    if args.validate:
        validate()
        return

    if not args.dataset:
        parser.error("--dataset is required (use --validate to see options)")

    if args.submit:
        submit_job(args)
        return

    # Local execution (needs pandas + pyarrow installed)
    run_download(args.dataset, args.region, args.max_samples, args.split,
                 args.document_field, args.prompt_template, args.hf_config)


if __name__ == "__main__":
    main()
