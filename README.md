# Aegis Platform
* Aegis is a multi‑cloud, multi‑hardware ML/AI platform for end‑to‑end model lifecycle: data → RAG → training (DeepSpeed, TPU/TPU‑like support) → RLHF → serving (vLLM / Triton) → observability & governance. It runs on AWS / Azure / GCP / OCI / Alibaba, supports GPU / CPU / TPU, does large‑scale distributed training and high‑throughput inference, and includes both generative AI pipelines and agentic automation (policy‑governed actions, automated approvals, compensation/rollback). The platform provides CI/CD workflows, security (Vault/KMS, short‑lived tokens, JWKS/mTLS examples), policy enforcement (OPA/rego), auditor tooling (decision_log → Metabase, Auditor UI), and robust chaos/rollback tests for operational safety.

# Key Features

   *  Multi‑cloud deploys: AWS / Azure / GCP / OCI / Alibaba.
   * Hardware support: GPU, CPU, TPU.
   * Large‑scale training: DeepSpeed + sharding, TPU support.
   * Serving: vLLM, Triton, batching/quantization support.
   * Generative pipelines: RAG, tokenizers, RLHF scaffolds.
   * Agentic flows: OPA policies, auto‑approval, time‑boxed signoffs, authenticated executor.
   * Security & secrets: Vault + KMS rotation automations, short‑lived JWTs, JWKS endpoint, revocations.
   * Observability & SLOs: Prometheus rules, alerts, compensation controller, Metabase dashboards.
   * Resilience testing: chaos matrix, rollback validation, automated DR rehearsals.
   * Auditor UX: OIDC + RBAC Auditor UI, decision_log trail, saved queries.




