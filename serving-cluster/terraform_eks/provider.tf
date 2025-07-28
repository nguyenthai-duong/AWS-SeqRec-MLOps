provider "aws" {
  region = var.aws_region
  # Đảm bảo credentials được cấu hình qua ~/.aws/credentials hoặc biến môi trường
}

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}