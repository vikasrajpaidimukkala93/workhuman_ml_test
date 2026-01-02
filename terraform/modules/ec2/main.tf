resource "aws_instance" "app_server" {
  ami           = var.ami_id
  instance_type = var.instance_type
  key_name      = var.key_name

  vpc_security_group_ids = var.vpc_security_group_ids
  subnet_id              = var.subnet_id

  tags = {
    Name        = "${var.project_name}-${var.environment}-api"
    Environment = var.environment
  }

  user_data = <<-EOF
              #!/bin/bash
              # Custom user data script (e.g., install Docker, run container)
              EOF
}
