provider "aws" {
  region = "af-south-1"
}

data "aws_security_group" "existing_sg" {
  name = "launch-wizard-1"  # Replace with your security group name
}

# Define EC2 instance
resource "aws_instance" "app_server" {
  ami           = "ami-0b247392537b9d99d" # Amazon Linux 2 AMI
  instance_type = "t3.micro"

  key_name = var.key_name  # SSH key

  vpc_security_group_ids = [data.aws_security_group.existing_sg.id]

  user_data = <<-EOF
              #!/bin/bash
              sudo yum update -y

              # Install Docker
              sudo yum install docker -y
              sudo service docker start
              sudo usermod -a -G docker ec2-user

              # Pull the Docker image from Docker Hub
              docker login -u "${var.DOCKER_USERNAME}" -p "${var.DOCKER_PASSWORD}"
              docker pull ${var.DOCKER_USERNAME}/aurora-server:latest

              # Run the Docker container
              docker run -d -p 80:8080 ${var.DOCKER_USERNAME}/aurora-server:latest
              EOF

  # Specify the root block device size
  root_block_device {
    volume_size = 20  # Set the total size to 20 GB
    volume_type = "gp2"  # General Purpose SSD (optional)
  }

  tags = {
    Name = "AuroraServer"
  }

  lifecycle {
    create_before_destroy = true  # Create new instance before destroying old one
  }
}

# Associate an existing Elastic IP with the instance
resource "aws_eip_association" "eip_assoc" {
  instance_id   = aws_instance.app_server.id
  allocation_id = "eipalloc-0adaebc65e5e4e55b"  # Use your actual Elastic IP allocation ID
}

output "instance_ip" {
  value = aws_instance.app_server.public_ip
}
