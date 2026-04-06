# ── Default VPC + Subnet (auto-lookup, no hardcoding) ────────────────────────

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

# ── Latest Amazon Linux 2023 AMI ─────────────────────────────────────────────

data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ── Security Group — outbound only ───────────────────────────────────────────

resource "aws_security_group" "retrain" {
  name        = "${var.project_name}-retrain"
  description = "EfficientNet retrain spot instance - outbound only"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "Allow all outbound (Docker pull, S3, MLflow, Supabase)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

# ── IAM Role for the EC2 instance ────────────────────────────────────────────

resource "aws_iam_role" "retrain_instance" {
  name = "${var.project_name}-retrain-instance"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "retrain_s3" {
  name = "flywheel-s3-readwrite"
  role = aws_iam_role.retrain_instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ListBucket"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = aws_s3_bucket.flywheel.arn
      },
      {
        Sid    = "ReadWriteObjects"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:HeadObject"]
        Resource = "${aws_s3_bucket.flywheel.arn}/*"
      }
    ]
  })
}

resource "aws_iam_role_policy" "retrain_self_terminate" {
  name = "self-terminate"
  role = aws_iam_role.retrain_instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid      = "SelfTerminate"
      Effect   = "Allow"
      Action   = "ec2:TerminateInstances"
      Resource = "arn:aws:ec2:${var.aws_region}:*:instance/*"
      Condition = {
        StringEquals = {
          "ec2:ResourceTag/ManagedBy" = "terraform"
          "ec2:ResourceTag/Project"   = var.project_name
        }
      }
    }]
  })
}

resource "aws_iam_instance_profile" "retrain" {
  name = "${var.project_name}-retrain"
  role = aws_iam_role.retrain_instance.name
}

# ── Launch Template ───────────────────────────────────────────────────────────
# User data (with secrets) is injected by GitHub Actions at launch time —
# not stored here so secrets never land in Terraform state.

resource "aws_launch_template" "retrain" {
  name        = "${var.project_name}-retrain"
  description = "EfficientNet retrain on EC2 Spot — launched by GitHub Actions"

  image_id      = data.aws_ami.al2023.id
  instance_type = var.retrain_instance_type

  iam_instance_profile {
    name = aws_iam_instance_profile.retrain.name
  }

  vpc_security_group_ids = [aws_security_group.retrain.id]

  # Allow Docker containers to reach IMDSv2 (hop limit = 2)
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
    http_endpoint               = "enabled"
  }

  instance_market_options {
    market_type = "spot"
    spot_options {
      instance_interruption_behavior = "terminate"
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags = merge(local.common_tags, {
      Name    = "${var.project_name}-retrain"
      Purpose = "effnet-retrain"
    })
  }

  tags = local.common_tags
}
