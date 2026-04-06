terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Partial backend — bucket and region are passed at init time via -backend-config.
  # See terraform/backend.hcl.example for the template.
  # Native S3 locking (Terraform 1.10+) — no DynamoDB needed.
  backend "s3" {
    key          = "terraform/state/terraform.tfstate"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Flywheel S3 bucket ────────────────────────────────────────────────────────

resource "aws_s3_bucket" "flywheel" {
  bucket = var.s3_bucket_name

  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "flywheel" {
  bucket = aws_s3_bucket.flywheel.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "flywheel" {
  bucket = aws_s3_bucket.flywheel.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "flywheel" {
  bucket = aws_s3_bucket.flywheel.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "flywheel" {
  bucket = aws_s3_bucket.flywheel.id

  rule {
    id     = "transition-crops-to-ia"
    status = "Enabled"

    filter {
      prefix = "${var.s3_prefix}/crops/"
    }

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 365
      storage_class = "GLACIER_IR"
    }
  }

  rule {
    id     = "transition-labels-to-ia"
    status = "Enabled"

    filter {
      prefix = "${var.s3_prefix}/labels/"
    }

    transition {
      days          = 180
      storage_class = "STANDARD_IA"
    }
  }
}

# ── Locals ────────────────────────────────────────────────────────────────────

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
