terraform {
  required_providers {
    fly = {
      source  = "fly-apps/fly"
      version = "~> 0.0.23"
    }
  }
}

provider "fly" {
  fly_api_token = var.fly_api_token
}

resource "fly_app" "fastapi_app" {
  name = var.app_name
  org  = var.fly_org
}

resource "fly_ip" "fastapi_ip" {
  app  = fly_app.fastapi_app.name
  type = "v4"
}

resource "fly_ip" "fastapi_ip_v6" {
  app  = fly_app.fastapi_app.name
  type = "v6"
}

resource "fly_volume" "fastapi_data" {
  name   = "fastapi_data"
  app    = fly_app.fastapi_app.name
  size   = 1
  region = var.fly_region
}

resource "fly_machine" "fastapi_machine" {
  app    = fly_app.fastapi_app.name
  region = var.fly_region
  name   = "fastapi-machine"
  image  = "registry.fly.io/${fly_app.fastapi_app.name}:latest"
  services = [
    {
      ports = [
        {
          port     = 443
          handlers = ["tls", "http"]
        },
        {
          port     = 80
          handlers = ["http"]
        }
      ]
      "protocol" : "tcp",
      "internal_port" : 8080
    }
  ]
  cpus = 1
  memorymb = 256

  env = {
      DATABASE_URL = var.db_url
      DEBUG = "0"
      GROQ_API = var.groq_api
      # Add any other environment variables your app needs
    }

  mounts = [
    {
      path   = "/data"
      volume = fly_volume.fastapi_data.id
    }
  ]
}

variable "db_url" {
  type = stirng
  description = "DB url"
  }

variable "groq_api" {
  type = stirng
  description = "GROQ API"
}

variable "fly_api_token" {
  type        = string
  description = "Fly.io API token"
}

variable "app_name" {
  type        = string
  description = "Name of the Fly.io app"
}

variable "fly_org" {
  type        = string
  description = "Fly.io organization"
}

variable "fly_region" {
  type        = string
  description = "Fly.io region"
  default     = "lax"
}

output "app_name" {
  value = fly_app.fastapi_app.name
}

output "public_ip" {
  value = fly_ip.fastapi_ip.address
}

output "public_ip_v6" {
  value = fly_ip.fastapi_ip_v6.address
}
