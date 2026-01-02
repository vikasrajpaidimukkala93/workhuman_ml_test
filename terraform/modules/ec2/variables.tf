variable "project_name" {}
variable "environment" {}
variable "ami_id" {}
variable "instance_type" {
  default = "t2.micro"
}
variable "key_name" {
    default = null
}
variable "vpc_security_group_ids" {
  type = list(string)
}
variable "subnet_id" {
    default = null
}
