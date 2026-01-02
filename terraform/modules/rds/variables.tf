variable "project_name" {}
variable "environment" {}
variable "instance_class" {
  default = "db.t3.micro"
}
variable "db_name" {}
variable "username" {}
variable "password" {}
variable "vpc_security_group_ids" {
  type = list(string)
}
variable "db_subnet_group_name" {
    default = null
}
