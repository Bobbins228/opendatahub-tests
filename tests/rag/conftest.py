import os
from typing import Any, Dict, Generator

import portforward
import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from llama_stack_client import LlamaStackClient, APIConnectionError
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.service import Service
from ocp_resources.config_map import ConfigMap
from .utils import get_etcd_deployment_template, get_milvus_deployment_template
from ocp_resources.namespace import Namespace
from ocp_resources.project_project_openshift_io import Project
from simple_logger.logger import get_logger
from timeout_sampler import retry

from utilities.constants import DscComponents, Timeout
from utilities.data_science_cluster_utils import update_components_in_dsc
from utilities.general import generate_random_name
from utilities.infra import create_ns
from ocp_resources.llama_stack_distribution import LlamaStackDistribution
from utilities.rag_utils import create_llama_stack_distribution

LOGGER = get_logger(name=__name__)


def llama_stack_server() -> Dict[str, Any]:
    rag_vllm_url = os.getenv("RAG_VLLM_URL")
    rag_vllm_model = os.getenv("RAG_VLLM_MODEL")
    rag_vllm_token = os.getenv("RAG_VLLM_TOKEN")

    return {
        "containerSpec": {
            "resources": {
                "requests": {"cpu": "250m", "memory": "500Mi"},
                "limits": {"cpu": "2", "memory": "12Gi"},
            },
            "env": [
                {"name": "INFERENCE_MODEL", "value": rag_vllm_model},
                {"name": "VLLM_TLS_VERIFY", "value": "false"},
                {"name": "VLLM_API_TOKEN", "value": rag_vllm_token},
                {"name": "VLLM_URL", "value": rag_vllm_url},
                {"name": "FMS_ORCHESTRATOR_URL", "value": "http://localhost"},
                {"name": "MILVUS_DB_PATH", "value": "~/.llama/distributions/rh/milvus.db"},
                {"name": "MILVUS_ENDPOINT", "value": "http://rag-milvus-service:19530"},
                {"name": "MILVUS_TOKEN", "value": "root:Milvus"},
            ],
            "name": "llama-stack",
            "port": 8321,
        },
        "distribution": {"image": "quay.io/opendatahub/llama-stack:odh"},
        "userConfig": {"configMapName": "rag-llama-stack-config-map"},
    }


@pytest.fixture(scope="class")
def enabled_llama_stack_operator(dsc_resource: DataScienceCluster) -> Generator[DataScienceCluster, Any, Any]:
    with update_components_in_dsc(
        dsc=dsc_resource,
        components={
            DscComponents.LLAMASTACKOPERATOR: DscComponents.ManagementState.MANAGED,
        },
        wait_for_components_state=True,
    ) as dsc:
        yield dsc


@pytest.fixture(scope="class")
def rag_test_namespace(
    admin_client: DynamicClient, unprivileged_client: DynamicClient
) -> Generator[Namespace | Project, Any, Any]:
    namespace_name = generate_random_name(prefix="rag-test")
    with create_ns(name=namespace_name, admin_client=admin_client, unprivileged_client=unprivileged_client) as ns:
        yield ns


@pytest.fixture(scope="class")
def etcd_deployment(
    rag_test_namespace: Namespace | Project,
    admin_client: DynamicClient,
) -> Generator[Deployment, Any, Any]:
    with Deployment(
        client=admin_client,
        namespace=rag_test_namespace.name,
        name="rag-etcd-deployment",
        replicas=1,
        selector={"matchLabels": {"app": "etcd"}},
        strategy={"type": "Recreate"},
        template=get_etcd_deployment_template(),
        teardown=True,
    ) as deployment:
        deployment.wait_for_replicas(deployed=True, timeout=Timeout.TIMEOUT_2MIN)
        yield deployment


@pytest.fixture(scope="class")
def etcd_service(admin_client: DynamicClient, rag_test_namespace: Namespace | Project) -> Generator[Service, Any, Any]:
    with Service(
        client=admin_client,
        namespace=rag_test_namespace.name,
        name="rag-etcd-service",
        ports=[
            {
                "port": 2379,
                "targetPort": 2379,
            }
        ],
        selector={"app": "etcd"},
    ) as service:
        yield service


@pytest.fixture(scope="class")
def remote_milvus_deployment(
    rag_test_namespace: Namespace | Project,
    admin_client: DynamicClient,
    etcd_deployment: Deployment,
    etcd_service: Service,
) -> Generator[Deployment, Any, Any]:
    with Deployment(
        client=admin_client,
        namespace=rag_test_namespace.name,
        name="rag-milvus-deployment",
        replicas=1,
        selector={"matchLabels": {"app": "milvus-standalone"}},
        strategy={"type": "Recreate"},
        template=get_milvus_deployment_template(),
        teardown=True,
    ) as deployment:
        deployment.wait_for_replicas(deployed=True, timeout=Timeout.TIMEOUT_2MIN)
        yield deployment


@pytest.fixture(scope="class")
def milvus_service(
    admin_client: DynamicClient, rag_test_namespace: Namespace | Project
) -> Generator[Service, Any, Any]:
    with Service(
        client=admin_client,
        namespace=rag_test_namespace.name,
        name="rag-milvus-service",
        ports=[
            {
                "name": "grpc",
                "port": 19530,
                "targetPort": 19530,
            },
        ],
        selector={"app": "milvus-standalone"},
    ) as service:
        yield service


@pytest.fixture(scope="class")
def llama_stack_config_map(
    rag_test_namespace: Namespace | Project,
    admin_client: DynamicClient,
) -> Generator[ConfigMap, Any, Any]:
    with ConfigMap(
        client=admin_client,
        namespace=rag_test_namespace.name,
        name="rag-llama-stack-config-map",
        data={
            "run.yaml": """# Llama Stack Configuration
version: "2"
image_name: rh
apis:
  - agents
  - datasetio
  - eval
  - inference
  - safety
  - scoring
  - telemetry
  - tool_runtime
  - vector_io
providers:
  inference:
    - provider_id: vllm-inference
      provider_type: remote::vllm
      config:
        url: ${env.VLLM_URL:=http://localhost:8000/v1}
        max_tokens: ${env.VLLM_MAX_TOKENS:=4096}
        api_token: ${env.VLLM_API_TOKEN:=fake}
        tls_verify: ${env.VLLM_TLS_VERIFY:=true}
    - provider_id: sentence-transformers
      provider_type: inline::sentence-transformers
      config: {}
  vector_io:
    - provider_id: remote-milvus
      provider_type: remote::milvus
      config:
        uri: ${env.MILVUS_ENDPOINT:=http://localhost:19530}
        token: ${env.MILVUS_TOKEN:=root:Milvus}
        kvstore:
          type: sqlite
          db_path: ~/.llama/distributions/rh/milvus_remote_registry.db
  safety:
    - provider_id: trustyai_fms
      provider_type: remote::trustyai_fms
      config:
        orchestrator_url: ${env.FMS_ORCHESTRATOR_URL:=}
        ssl_cert_path: ${env.FMS_SSL_CERT_PATH:=}
        shields: {}
  agents:
    - provider_id: meta-reference
      provider_type: inline::meta-reference
      config:
        persistence_store:
          type: sqlite
          namespace: null
          db_path: /opt/app-root/src/.llama/distributions/rh/agents_store.db
        responses_store:
          type: sqlite
          db_path: /opt/app-root/src/.llama/distributions/rh/responses_store.db
  eval:
    - provider_id: trustyai_lmeval
      provider_type: remote::trustyai_lmeval
      config:
        use_k8s: True
        base_url: ${env.VLLM_URL:=http://localhost:8000/v1}
  datasetio:
    - provider_id: huggingface
      provider_type: remote::huggingface
      config:
        kvstore:
          type: sqlite
          namespace: null
          db_path: /opt/app-root/src/.llama/distributions/rh/huggingface_datasetio.db
    - provider_id: localfs
      provider_type: inline::localfs
      config:
        kvstore:
          type: sqlite
          namespace: null
          db_path: /opt/app-root/src/.llama/distributions/rh/localfs_datasetio.db
  scoring:
    - provider_id: basic
      provider_type: inline::basic
      config: {}
    - provider_id: llm-as-judge
      provider_type: inline::llm-as-judge
      config: {}
    - provider_id: braintrust
      provider_type: inline::braintrust
      config:
        openai_api_key: ${env.OPENAI_API_KEY:=}
  telemetry:
    - provider_id: meta-reference
      provider_type: inline::meta-reference
      config:
        service_name: "${env.OTEL_SERVICE_NAME:=}"
        sinks: console,sqlite
        sqlite_db_path: /opt/app-root/src/.llama/distributions/rh/trace_store.db
        otel_exporter_otlp_endpoint: ${env.OTEL_EXPORTER_OTLP_ENDPOINT:=}
  tool_runtime:
    - provider_id: brave-search
      provider_type: remote::brave-search
      config:
        api_key: ${env.BRAVE_SEARCH_API_KEY:=}
        max_results: 3
    - provider_id: tavily-search
      provider_type: remote::tavily-search
      config:
        api_key: ${env.TAVILY_SEARCH_API_KEY:=}
        max_results: 3
    - provider_id: rag-runtime
      provider_type: inline::rag-runtime
      config: {}
    - provider_id: model-context-protocol
      provider_type: remote::model-context-protocol
      config: {}
metadata_store:
  type: sqlite
  db_path: /opt/app-root/src/.llama/distributions/rh/registry.db
inference_store:
  type: sqlite
  db_path: /opt/app-root/src/.llama/distributions/rh/inference_store.db
models:
  - metadata: {}
    model_id: ${env.INFERENCE_MODEL}
    provider_id: vllm-inference
    model_type: llm
  - metadata:
      embedding_dimension: 768
    model_id: granite-embedding-125m
    provider_id: sentence-transformers
    provider_model_id: ibm-granite/granite-embedding-125m-english
    model_type: embedding
shields: []
vector_dbs: []
datasets: []
scoring_fns: []
benchmarks: []
tool_groups:
  - toolgroup_id: builtin::websearch
    provider_id: tavily-search
  - toolgroup_id: builtin::rag
    provider_id: rag-runtime
server:
  port: 8321
  external_providers_dir: /opt/app-root/.llama/providers.d"""
        },
    ) as config_map:
        yield config_map


@pytest.fixture(scope="class")
def llama_stack_distribution_from_template(
    enabled_llama_stack_operator: Generator[DataScienceCluster, Any, Any],
    rag_test_namespace: Namespace | Project,
    request: FixtureRequest,
    admin_client: DynamicClient,
) -> Generator[LlamaStackDistribution, Any, Any]:
    with create_llama_stack_distribution(
        client=admin_client,
        name="rag-llama-stack-distribution",
        namespace=rag_test_namespace.name,
        replicas=1,
        server=llama_stack_server(),
    ) as llama_stack_distribution:
        llama_stack_distribution.wait_for_condition(condition="HealthCheck", status="True", timeout=240)
        yield llama_stack_distribution


@pytest.fixture(scope="class")
def llama_stack_distribution_deployment(
    rag_test_namespace: Namespace | Project,
    admin_client: DynamicClient,
    llama_stack_distribution_from_template: Generator[LlamaStackDistribution, Any, Any],
) -> Generator[Deployment, Any, Any]:
    deployment = Deployment(
        client=admin_client,
        namespace=rag_test_namespace.name,
        name="rag-llama-stack-distribution",
    )

    deployment.wait(timeout=Timeout.TIMEOUT_2MIN)
    yield deployment


@retry(wait_timeout=Timeout.TIMEOUT_1MIN, sleep=5)
def wait_for_llama_stack_ready(client: LlamaStackClient) -> bool:
    try:
        client.inspect.health()
        version = client.inspect.version()
        LOGGER.info(f"Llama Stack server (v{version.version}) is available!")
        return True
    except APIConnectionError as e:
        LOGGER.debug(f"Llama Stack server not ready yet: {e}")
        return False
    except Exception as e:
        LOGGER.warning(f"Unexpected error checking Llama Stack readiness: {e}")
        return False


@pytest.fixture(scope="class")
def rag_lls_client(
    admin_client: DynamicClient,
    rag_test_namespace: Namespace | Project,
    llama_stack_distribution_deployment: Deployment,
) -> Generator[LlamaStackClient, Any, Any]:
    """
    Returns a ready to use LlamaStackClient,  enabling port forwarding
    from the llama-stack-server service:8321 to localhost:8321

    Args:
        admin_client (DynamicClient): Kubernetes dynamic client for cluster operations
        rag_test_namespace (Namespace | Project): Namespace or project containing RAG test resources
        llama_stack_distribution_deployment (Deployment): LlamaStack distribution deployment resource

    Yields:
        Generator[LlamaStackClient, Any, Any]: Configured LlamaStackClient for RAG testing
    """
    try:
        with portforward.forward(
            pod_or_service="rag-llama-stack-distribution-service",
            namespace=rag_test_namespace.name,
            from_port=8321,
            to_port=8321,
            waiting=15,
        ):
            client = LlamaStackClient(
                base_url="http://localhost:8321",
                timeout=120.0,
            )
            wait_for_llama_stack_ready(client=client)
            yield client
    except Exception as e:
        LOGGER.error(f"Failed to set up port forwarding: {e}")
        raise
