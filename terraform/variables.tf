variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short project identifier — used as a prefix for resource names."
  type        = string
  default     = "ppe-detection"
}

variable "environment" {
  description = "Deployment environment tag (e.g. production, staging)."
  type        = string
  default     = "production"
}

variable "s3_bucket_name" {
  description = "Name of the existing flywheel S3 bucket. Must match the import block id."
  type        = string
  default     = "ppe-flywheel"
}

variable "s3_prefix" {
  description = "Key prefix inside the flywheel bucket (matches S3_PREFIX env var used by the app)."
  type        = string
  default     = "ppe-flywheel"
}

variable "retrain_instance_type" {
  description = "EC2 instance type for EfficientNet retrain spot job."
  type        = string
  default     = "t3.medium"
}
