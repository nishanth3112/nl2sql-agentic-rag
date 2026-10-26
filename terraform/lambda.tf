# NOTE on deploy order: this references an image tag in the ECR repo
# defined in ecr.tf. That image doesn't exist until you've built and pushed
# it -- see terraform/README.md for the required two-phase apply (create
# the ECR repo first, push the image, then apply the rest).
resource "aws_lambda_function" "api" {
  function_name = "${var.project_name}-api"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api.repository_url}:${var.lambda_image_tag}"

  # Generous timeout: a single request can involve up to 3 LLM calls
  # (planner, generator, classifier) plus retries through the correction
  # loop, across three different providers with their own latency.
  timeout     = 60
  memory_size = 1024

  environment {
    variables = {
      GEMINI_API_KEY      = var.gemini_api_key
      GROQ_API_KEY        = var.groq_api_key
      OPENAI_API_KEY      = var.openai_api_key
      BENCHMARK_DATA_PATH  = "/var/task/data/bird-mini-dev"
      MAX_RETRIES         = tostring(var.max_retries)
      TRACING_ENABLED     = tostring(var.tracing_enabled)
      # Lambda's filesystem is read-only outside /tmp.
      TRACE_EXPORT_PATH   = "/tmp/spans.jsonl"
      PLANNER_PROVIDER    = var.planner_provider
      PLANNER_MODEL       = var.planner_model
      GENERATOR_PROVIDER  = var.generator_provider
      GENERATOR_MODEL     = var.generator_model
      CLASSIFIER_PROVIDER = var.classifier_provider
      CLASSIFIER_MODEL    = var.classifier_model
      # REDIS_URL intentionally omitted: no managed Redis is provisioned
      # here (ElastiCache has an hourly cost with no free tier) -- the
      # cache module already no-ops gracefully when Redis is unreachable.
    }
  }

  depends_on = [aws_iam_role_policy_attachment.lambda_basic]
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${aws_lambda_function.api.function_name}"
  retention_in_days = 14 # keeps CloudWatch Logs storage cost near zero
}
