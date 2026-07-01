"""
Reads CloudFormation stack outputs after cdk deploy
and saves them to S3 as infra_outputs.json
(so Pipeline 2 can read them the same way as before)
"""
import boto3
import json
import os


def save_outputs():
    region   = os.environ.get("AWS_REGION", "us-east-1")
    account  = boto3.client("sts", region_name=region)\
                    .get_caller_identity()["Account"]
    bucket   = f"sagemaker-adult-income-cdk-{account}"

    cf_client = boto3.client("cloudformation", region_name=region)
    s3_client = boto3.client("s3",             region_name=region)

    # Read CloudFormation stack outputs
    response = cf_client.describe_stacks(StackName="AdultIncomeSageMakerStack")
    outputs  = {
        o["OutputKey"]: o["OutputValue"]
        for o in response["Stacks"][0].get("Outputs", [])
    }

    print(f"✅ CDK Stack Outputs:")
    for k, v in outputs.items():
        print(f"   {k}: {v}")

    # Map to the same format Pipeline 2 expects
    infra_outputs = {
        "domain_id"   : outputs["DomainId"],
        "domain_name" : "adult-income-sagemaker-studio",
        "role_arn"    : outputs["RoleArn"],
        "bucket_name" : outputs["BucketName"],
        "profile_name": "adult-income-user",
        "region"      : region,
        "account"     : account
    }

    # Save locally
    with open("infra_outputs.json", "w") as f:
        json.dump(infra_outputs, f, indent=2)

    # Save to S3 (same key as before — Pipeline 2 reads from here)
    s3_client.put_object(
        Bucket=bucket,
        Key="infra/infra_outputs.json",
        Body=json.dumps(infra_outputs, indent=2)
    )

    print(f"\n✅ Infra outputs saved!")
    print(f"   Domain ID  : {infra_outputs['domain_id']}")
    print(f"   Role ARN   : {infra_outputs['role_arn']}")
    print(f"   Bucket     : {infra_outputs['bucket_name']}")

    return infra_outputs


if __name__ == "__main__":
    save_outputs()