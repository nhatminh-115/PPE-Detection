output "s3_bucket_name" {
  description = "Flywheel S3 bucket name — set as S3_BUCKET in .env and GitHub secrets."
  value       = aws_s3_bucket.flywheel.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the flywheel S3 bucket."
  value       = aws_s3_bucket.flywheel.arn
}

output "retrain_launch_template_id" {
  description = "Launch Template ID for the retrain spot instance — used by GitHub Actions."
  value       = aws_launch_template.retrain.id
}

output "retrain_subnet_id" {
  description = "Subnet ID used to launch the retrain spot instance."
  value       = data.aws_subnets.default.ids[0]
}

