# Data source for existing S3 bucket
data "aws_s3_bucket" "terraform_state" {
  bucket = "arcticlabs-server"
}

# Server-side encryption configuration for the existing bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = data.aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
