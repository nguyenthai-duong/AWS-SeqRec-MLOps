provider "aws" {
  region = var.aws_region
}

data "aws_vpc" "default" {
  default = true
}

# <-- sửa đây -->
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

resource "aws_security_group" "docker_compose_sg" {
  name        = "docker-compose-sg"
  description = "Allow SSH & all inbound for test"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 0
    to_port     = 65535
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

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "datn-ec2-profile"
  role = "datn-ec2"
}

resource "aws_instance" "docker_compose" {
  ami                         = data.aws_ami.amazon_linux.id
  instance_type               = var.instance_type
  subnet_id                   = data.aws_subnets.default.ids[0]
  key_name                    = var.key_name
  associate_public_ip_address = true
  vpc_security_group_ids      = [aws_security_group.docker_compose_sg.id]
  iam_instance_profile        = aws_iam_instance_profile.ec2_profile.name

  user_data = <<-EOF
    #!/bin/bash
    yum update -y

    # Cài docker
    yum install -y docker

    systemctl enable docker
    systemctl start docker
    usermod -aG docker ec2-user

    # Cài awscli
    yum install -y awscli

    # Sync airflow folder từ S3
    mkdir -p /home/ec2-user/app/${var.s3_prefix_airflow}
    aws s3 sync s3://${var.bucket_name}/${var.s3_prefix_airflow}/ /home/ec2-user/app/${var.s3_prefix_airflow}
    chown -R ec2-user:ec2-user /home/ec2-user/app

    # Cài docker compose plugin bằng tay (dùng đúng với Docker bản mới)
    su - ec2-user -c '
      export DOCKER_CONFIG=$${DOCKER_CONFIG:-$${HOME}/.docker}
      mkdir -p $DOCKER_CONFIG/cli-plugins
      curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
      chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
    '

    # Ghi systemd service cho Docker Compose
    cat <<EOL > /etc/systemd/system/docker-compose-app.service
    [Unit]
    Description=Docker Compose App
    Requires=docker.service
    After=docker.service

    [Service]
    Type=oneshot
    RemainAfterExit=true
    User=ec2-user
    WorkingDirectory=/home/ec2-user/app/${var.s3_prefix_airflow}
    ExecStart=/usr/bin/docker compose up -d
    ExecStop=/usr/bin/docker compose down

    [Install]
    WantedBy=multi-user.target
    EOL

    systemctl daemon-reload
    systemctl enable docker-compose-app
    systemctl start docker-compose-app
  EOF


  tags = {
    Name = "docker-compose-airflow"
  }
}
