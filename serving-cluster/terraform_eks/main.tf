# VPC cho EKS
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true

  tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.29"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true
  
  # Kích hoạt CloudWatch logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  eks_managed_node_groups = {
    default = {
      instance_types = [var.node_instance_type]
      min_size       = 1
      max_size       = 5
      desired_size   = var.desired_capacity
      disk_size      = 30
    }
  }

  tags = {
    Environment = "production"
  }
}

resource "aws_iam_role_policy_attachment" "autoscaler" {
  for_each = module.eks.eks_managed_node_groups

  role       = each.value.iam_role_name
  policy_arn = "arn:aws:iam::aws:policy/AutoScalingFullAccess"
}

# IAM Role cho pod truy cập tất cả RDS, S3, DynamoDB
resource "aws_iam_role" "pod_role" {
  name = "${var.cluster_name}-pod-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${module.eks.oidc_provider}:sub" = "system:serviceaccount:*:*"
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "pod_policy" {
  name = "${var.cluster_name}-pod-policy"
  role = aws_iam_role.pod_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "rds-db:connect"
        ]
        Resource = [
          "arn:aws:rds:${var.aws_region}:${data.aws_caller_identity.current.account_id}:db:*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::*",
          "arn:aws:s3:::*/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = [
          "arn:aws:dynamodb:${var.aws_region}:${data.aws_caller_identity.current.account_id}:table/*"
        ]
      }
    ]
  })
}

# Dữ liệu AWS account ID
data "aws_caller_identity" "current" {}

# Cấu hình Kubernetes provider
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", var.cluster_name]
  }
}

# Tạo ServiceAccount và gắn IAM Role
resource "kubernetes_service_account" "pod_sa" {
  metadata {
    name      = "pod-sa"
    namespace = "default"
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.pod_role.arn
    }
  }
}


# Monitor
resource "aws_iam_role" "cloudwatch_agent_role" {
  name = "${var.cluster_name}-cloudwatch-agent-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Principal = {
        Federated = module.eks.oidc_provider_arn
      },
      Action = "sts:AssumeRoleWithWebIdentity",
      Condition = {
        StringEquals = {
          "${module.eks.oidc_provider}:sub" = "system:serviceaccount:amazon-cloudwatch:cloudwatch-agent"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "cloudwatch_agent_fullaccess_attach" {
  role       = aws_iam_role.cloudwatch_agent_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchFullAccess"
}

resource "kubernetes_namespace" "cloudwatch" {
  metadata {
    name = "amazon-cloudwatch"
  }
}

resource "kubernetes_service_account" "cloudwatch_agent" {
  metadata {
    name      = "cloudwatch-agent"
    namespace = kubernetes_namespace.cloudwatch.metadata[0].name
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.cloudwatch_agent_role.arn
    }
  }
}

