pipeline {
    agent any

    stages {
        stage('Load Environment Variables') {
            steps {
                script {
                    env.ACTUAL_BRANCH = env.BRANCH_NAME ?: env.GIT_BRANCH ?: "unknown"
                    echo "Detected Branch: ${env.ACTUAL_BRANCH}"

                    withCredentials([
                        string(credentialsId: 'DEBUG', variable: 'DEBUG'),
                        string(credentialsId: 'DB_PROD_HOST', variable: 'DB_PROD_HOST'),
                        string(credentialsId: 'DB_PROD_NAME', variable: 'DB_PROD_NAME'),
                        string(credentialsId: 'DB_PROD_USER', variable: 'DB_PROD_USER'),
                        string(credentialsId: 'DB_PROD_PASSWORD', variable: 'DB_PROD_PASS'),
                        string(credentialsId: 'DB_PROD_PORT', variable: 'DB_PROD_PORT')
            ]) {
                if (env.ACTUAL_BRANCH.contains('main')) {
                    echo "Loading Production Specific Credentials..."
                    withCredentials([
                        string(credentialsId: 'HH_PROD_DB_HOST', variable: 'HH_PROD_DB_HOST'),
                        string(credentialsId: 'HH_PROD_DB_NAME', variable: 'HH_PROD_DB_NAME'),
                        string(credentialsId: 'HH_PROD_DB_USER', variable: 'HH_PROD_DB_USER'),
                        string(credentialsId: 'HH_PROD_DB_PASSWORD', variable: 'HH_PROD_DB_PASSWORD'),
                        string(credentialsId: 'HH_PROD_DB_PORT', variable: 'HH_PROD_DB_PORT')
                    ]) {
                        sh """
                        cat <<EOF > .env
                        
DEBUG=${DEBUG}
HH_DB_HOST=${HH_PROD_DB_HOST}
HH_DB_NAME=${HH_PROD_DB_NAME}
HH_DB_USER=${HH_PROD_DB_USER}
HH_DB_PASSWORD=${HH_PROD_DB_PASSWORD}
HH_DB_PORT=${HH_PROD_DB_PORT}
DB_HOST=${DB_PROD_HOST}
DB_NAME=${DB_PROD_NAME}
DB_USER=${DB_PROD_USER}
DB_PASSWORD=${DB_PROD_PASS}
DB_PORT=${DB_PROD_PORT}
EOF
                        """
                    }
                } else if (env.ACTUAL_BRANCH.contains('develop')) {
                    echo "Loading Development Specific Credentials..."
                    withCredentials([
                        string(credentialsId: 'HH_Dev_DB_HOST', variable: 'HH_Dev_DB_HOST'),
                        string(credentialsId: 'HH_DEV_DB_NAME', variable: 'HH_DEV_DB_NAME'),
                        string(credentialsId: 'HH_DEV_DB_USER', variable: 'HH_DEV_DB_USER'),
                        string(credentialsId: 'HH_DEV_DB_PASSWORD', variable: 'HH_DEV_DB_PASSWORD'),
                        string(credentialsId: 'HH_DEV_DB_PORT', variable: 'HH_DEV_DB_PORT')
                    ]) {
                        sh """
                        cat <<EOF > .env
DEBUG=${DEBUG}
HH_DB_HOST=${HH_Dev_DB_HOST}
HH_DB_NAME=${HH_DEV_DB_NAME}
HH_DB_USER=${HH_DEV_DB_USER}
HH_DB_PASSWORD=${HH_DEV_DB_PASSWORD}
HH_DB_PORT=${HH_DEV_DB_PORT}
DB_HOST=${DB_PROD_HOST}
DB_NAME=${DB_PROD_NAME}
DB_USER=${DB_PROD_USER}
DB_PASSWORD=${DB_PROD_PASS}
DB_PORT=${DB_PROD_PORT}
EOF
                                """
                            }
                        }
                    }
                }
            }
        }

        stage('Compress and Upload') {
            steps {
                script {
                    def currentBranch = env.BRANCH_NAME ?: env.GIT_BRANCH ?: "unknown"
                    
                    if (currentBranch.contains('main')) {
                        env.S3_BUCKET = "saladstop-prod-foodbowls"
                    } else {
                        env.S3_BUCKET = "foodbowls"
                    }
                    echo "Targeting S3 Bucket: ${env.S3_BUCKET}"
                    
                    sh "ls -la"
                    
                    sh """
                    tar -czf bowls.tar.gz *.py requirements.txt .env
                    aws s3 cp bowls.tar.gz s3://${env.S3_BUCKET}/bowls.tar.gz
                    echo "Successfully packaged *.py, requirements.txt, and .env into s3://${env.S3_BUCKET}/bowls.tar.gz"
                    """
                }
            }
        }

        stage('Cleanup SageMaker Resources') {
    steps {
        sh """
        case "${env.ACTUAL_BRANCH}" in
            *main*) RESOURCE_NAME="bowls" ;;
            *develop*) RESOURCE_NAME="bowls" ;;
            *) echo "Unknown branch: ${env.ACTUAL_BRANCH}. Skipping cleanup."; exit 0 ;;
        esac

        echo "Deleting endpoint \$RESOURCE_NAME if it exists..."
        aws sagemaker delete-endpoint --endpoint-name \$RESOURCE_NAME 2>/dev/null || echo "No existing endpoint to delete."
        aws sagemaker delete-endpoint-config --endpoint-config-name \$RESOURCE_NAME 2>/dev/null || echo "No existing endpoint-config to delete."

        echo "Waiting for endpoint \$RESOURCE_NAME to be fully deleted..."
        for i in \$(seq 1 30); do
            STATUS=\$(aws sagemaker describe-endpoint --endpoint-name \$RESOURCE_NAME --query 'EndpointStatus' --output text 2>/dev/null || echo "NOT_FOUND")
            echo "Current status: \$STATUS"
            if [ "\$STATUS" = "NOT_FOUND" ]; then
                echo "Endpoint fully deleted."
                break
            fi
            sleep 10
                done
                """
            }
        }

        stage('Deploy SageMaker Endpoint') {
            steps {
                script {
                    sh """
                    echo "Starting SageMaker deployment..."
                    aws s3 cp s3://${env.S3_BUCKET}/heybo_endpoint_deploy.py ./endpoint_deploy.py

                    export AWS_DEFAULT_REGION="ap-southeast-1"
                    export AWS_REGION="ap-southeast-1"

                    rm -rf venv
                    python3 -m venv venv --without-pip
                    . venv/bin/activate
                    curl -s https://bootstrap.pypa.io/get-pip.py | python3

                    ./venv/bin/pip install "sagemaker<3.0.0" boto3
                    ./venv/bin/python3 endpoint_deploy.py
                    """
                }
            }
        }
    }
}
