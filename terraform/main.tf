provider "aws" {
  region = var.aws_region
}

# --- Data Sources (Use default VPC) ---
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# --- Security Groups ---

resource "aws_security_group" "api_sg" {
  name        = "${var.project_name}-${var.environment}-api-sg"
  description = "Allow HTTP and SSH"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "rds_sg" {
  name        = "${var.project_name}-${var.environment}-rds-sg"
  description = "Allow Postgres from API SG"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.api_sg.id]
  }
}


# --- Modules ---

module "s3_data_bucket" {
  source      = "./modules/s3_bucket"
  bucket_name = "${var.project_name}-${var.environment}-data"
  environment = var.environment
}

module "rds_postgres" {
  source                 = "./modules/rds"
  project_name           = var.project_name
  environment            = var.environment
  db_name                = var.db_name
  username               = var.db_username
  password               = var.db_password
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
}

module "ec2_api" {
  source                 = "./modules/ec2"
  project_name           = var.project_name
  environment            = var.environment
  ami_id                 = var.ec2_ami_id
  instance_type          = var.ec2_instance_type
  vpc_security_group_ids = [aws_security_group.api_sg.id]
  # subnet_id not explicitly set, let AWS pick default
}

output "data_bucket_id" {
  value = module.s3_data_bucket.bucket_id
}

output "rds_endpoint" {
  value = module.rds_postgres.endpoint
}

output "api_public_ip" {
  value = module.ec2_api.public_ip
}
