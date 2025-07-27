output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = var.cluster_name
}

output "oidc_provider_arn" {
  description = "OIDC provider ARN"
  value       = module.eks.oidc_provider_arn
}