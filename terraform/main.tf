provider "aws" {
  region = var.aws_region
}

module "s3_data_bucket" {
  source      = "./modules/s3_bucket"
  bucket_name = "${var.project_name}-${var.environment}-data"
  environment = var.environment
}

output "data_bucket_id" {
  value = module.s3_data_bucket.bucket_id
}
