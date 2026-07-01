import os
import json
import boto3
import aws_cdk as cdk
from constructs import Construct

from aws_cdk import (
    Stack,
    RemovalPolicy,
    CfnOutput,
    aws_s3 as s3,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_events as events,
    aws_events_targets as targets,
    aws_sagemaker as sagemaker,
    aws_secretsmanager as secretsmanager,
)


class AdultIncomeSageMakerStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        github_repo: str,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        account = self.account
        region  = self.region

        # ─── Get VPC + Subnets via boto3 (avoids CDK synth-time lookup) ─
        ec2_client = boto3.client("ec2", region_name=os.environ.get("AWS_REGION", "us-east-1"))

        vpcs = ec2_client.describe_vpcs(
            Filters=[{"Name": "isDefault", "Values": ["true"]}]
        )
        vpc_id = vpcs["Vpcs"][0]["VpcId"]

        subnets = ec2_client.describe_subnets(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
        )
        subnet_ids = [s["SubnetId"] for s in subnets["Subnets"]][:2]

        print(f"✅ VPC ID   : {vpc_id}")
        print(f"✅ Subnets  : {subnet_ids}")

        # ─── 1. S3 Bucket ───────────────────────────────────
        bucket = s3.Bucket(
            self, "TrainingBucket",
            bucket_name=f"sagemaker-adult-income-pipeline-{account}",
            removal_policy=RemovalPolicy.RETAIN,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL
        )

        # ─── 2. SageMaker Execution Role ────────────────────
        sagemaker_role = iam.Role(
            self, "SageMakerExecutionRole",
            role_name="AdultIncomeSageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="SageMaker execution role for Adult Income pipeline",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonS3FullAccess"
                )
            ]
        )

        # ─── 3. SageMaker Studio Domain ─────────────────────
        domain = sagemaker.CfnDomain(
            self, "StudioDomain",
            domain_name="adult-income-sagemaker-studio",
            auth_mode="IAM",
            default_user_settings=sagemaker.CfnDomain.UserSettingsProperty(
                execution_role=sagemaker_role.role_arn,
                jupyter_server_app_settings=sagemaker.CfnDomain\
                    .JupyterServerAppSettingsProperty(
                        default_resource_spec=sagemaker.CfnDomain\
                            .ResourceSpecProperty(
                                instance_type="system"
                            )
                    ),
                kernel_gateway_app_settings=sagemaker.CfnDomain\
                    .KernelGatewayAppSettingsProperty(
                        default_resource_spec=sagemaker.CfnDomain\
                            .ResourceSpecProperty(
                                instance_type="ml.t3.medium"
                            )
                    )
            ),
            subnet_ids=subnet_ids,
            vpc_id=vpc_id,
            tags=[
                cdk.CfnTag(key="Project",     value="Adult-Income-Pipeline"),
                cdk.CfnTag(key="Environment", value="dev"),
                cdk.CfnTag(key="ManagedBy",   value="CDK")
            ]
        )

        # ─── 4. SageMaker User Profile ───────────────────────
        user_profile = sagemaker.CfnUserProfile(
            self, "StudioUserProfile",
            domain_id=domain.attr_domain_id,
            user_profile_name="adult-income-user",
            user_settings=sagemaker.CfnUserProfile.UserSettingsProperty(
                execution_role=sagemaker_role.role_arn
            )
        )
        user_profile.node.add_dependency(domain)

        # ─── 5. GitHub Token Secret Reference ───────────────
        github_token_secret = secretsmanager.Secret.from_secret_name_v2(
            self, "GitHubTokenSecret",
            "github-token"
        )

        # ─── 6. Lambda Execution Role ───────────────────────
        lambda_role = iam.Role(
            self, "LambdaDeployTriggerRole",
            role_name="LambdaAdultDeployTriggerRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Lambda role to trigger GitHub Actions (Adult Income)",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ]
        )

        github_token_secret.grant_read(lambda_role)

        # ─── 7. Lambda Function ─────────────────────────────
        lambda_code = f"""
import json
import urllib.request
import boto3

def lambda_handler(event, context):
    print(f"Event received: {{json.dumps(event)}}")

    detail = event.get("detail", {{}})
    status = detail.get("ModelApprovalStatus", "")

    print(f"Model Approval Status: {{status}}")

    if status != "Approved":
        print("Model not approved - skipping deploy")
        return {{"statusCode": 200, "body": "Not approved"}}

    sm_client = boto3.client("secretsmanager")
    secret    = sm_client.get_secret_value(SecretId="github-token")
    token     = secret["SecretString"]

    repo = "{github_repo}"
    url  = f"https://api.github.com/repos/{{repo}}/actions/workflows/deploy-pipeline.yml/dispatches"

    payload = json.dumps({{"ref": "master"}}).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={{
            "Authorization": f"Bearer {{token}}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        }},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as response:
            print(f"Pipeline 3 triggered! Status: {{response.status}}")
            return {{"statusCode": 200, "body": "Pipeline 3 triggered!"}}
    except Exception as e:
        print(f"Failed to trigger: {{str(e)}}")
        raise
"""

        deploy_trigger_fn = lambda_.Function(
            self, "DeployTriggerLambda",
            function_name="trigger-adult-deploy-pipeline",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_inline(lambda_code),
            role=lambda_role,
            timeout=cdk.Duration.seconds(30),
            description="Triggers GitHub Actions Pipeline 3 on Adult Income model approval"
        )

        # ─── 8. EventBridge Rule ─────────────────────────────
        model_approved_rule = events.Rule(
            self, "ModelApprovedRule",
            rule_name="adult-model-approved-rule",
            description="Trigger deploy pipeline when Adult Income model approved",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Model Package State Change"],
                detail={
                    "ModelApprovalStatus": ["Approved"],
                    "ModelPackageGroupName": ["AdultIncomeGroup"]
                }
            )
        )

        # ─── 9. Wire EventBridge → Lambda ────────────────────
        model_approved_rule.add_target(
            targets.LambdaFunction(deploy_trigger_fn)
        )

        # ─── 10. CloudFormation Outputs ──────────────────────
        CfnOutput(
            self, "DomainId",
            value=domain.attr_domain_id,
            description="SageMaker Studio Domain ID",
            export_name="AdultIncomeDomainId"
        )
        CfnOutput(
            self, "RoleArn",
            value=sagemaker_role.role_arn,
            description="SageMaker Execution Role ARN",
            export_name="AdultIncomeRoleArn"
        )
        CfnOutput(
            self, "BucketName",
            value=bucket.bucket_name,
            description="Training S3 Bucket Name",
            export_name="AdultIncomeBucketName"
        )
        CfnOutput(
            self, "LambdaArn",
            value=deploy_trigger_fn.function_arn,
            description="Deploy Trigger Lambda ARN",
            export_name="AdultIncomeLambdaArn"
        )