variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment (dev/prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "workhuman-ml"
}

variable "db_username" {
  description = "Database master username"
  type        = string
  default     = "workhuman"
}

variable "db_password" {
  description = "Database master password"
  type        = string
  sensitive   = true
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "workhuman_ml"
}

variable "ec2_instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t2.micro"
}

variable "ec2_ami_id" {
  description = "AMI ID for EC2 instance"
  type        = string
  # Example: Amazon Linux 2 AMI (HVM) - Kernel 5.10, SSD Volume Type
  default     = "ami-0cff7528ff583bf9a" 
}
