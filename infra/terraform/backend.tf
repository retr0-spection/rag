terraform {
  backend "s3" {
    bucket         = "arcticlabs-server-bucket"  # Replace with your S3 bucket name
    key            = "terraform/terraform.tfstate"  # The path to the state file within the bucket
    region         = "af-south-1"
    encrypt        = true  # Encrypt the state file
  }
}
