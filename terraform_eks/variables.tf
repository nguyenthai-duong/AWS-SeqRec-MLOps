variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-southeast-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "my-eks-cluster"
}

variable "node_instance_type" {
  description = "Instance type for EKS worker nodes"
  type        = string
  default     = "t3.xlarge"
}

variable "desired_capacity" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 1
}


