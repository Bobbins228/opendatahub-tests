

def get_milvus_deployment_template() -> dict[str, Any]:
    return {
        "metadata": {"labels": {"app": "milvus-standalone"}},
        "spec": {
            "containers": [
                {
                    "name": "milvus-standalone",
                    "image": "quay.io/mcampbel/milvus:v2.6.0",
                    "args": ["milvus", "run", "standalone"],
                    "ports": [{"containerPort": 19530, "protocol": "TCP"}],
                    "volumeMounts": [
                        {
                            "name": "milvus-data",
                            "mountPath": "/var/lib/milvus",
                        }
                    ],
                    "env": [
                        {"name": "DEPLOY_MODE", "value": "standalone"},
                        {"name": "ETCD_ENDPOINTS", "value": "rag-etcd-service:2379"},
                        {"name": "MINIO_ADDRESS", "value": ""},
                        {"name": "COMMON_STORAGETYPE", "value": "local"},
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "milvus-data",
                    "emptyDir": {},
                }
            ],
        },
    }


def get_etcd_deployment_template() -> dict[str, Any]:
    return {
        "metadata": {"labels": {"app": "etcd"}},
        "spec": {
            "containers": [
                {
                    "name": "etcd",
                    "image": "quay.io/coreos/etcd:v3.5.5",
                    "command": [
                        "etcd",
                        "--advertise-client-urls=http://127.0.0.1:2379",
                        "--listen-client-urls=http://0.0.0.0:2379",
                        "--data-dir=/etcd",
                    ],
                    "ports": [{"containerPort": 2379}],
                    "volumeMounts": [
                        {
                            "name": "etcd-data",
                            "mountPath": "/etcd",
                        }
                    ],
                    "env": [
                        {"name": "ETCD_AUTO_COMPACTION_MODE", "value": "revision"},
                        {"name": "ETCD_AUTO_COMPACTION_RETENTION", "value": "1000"},
                        {"name": "ETCD_QUOTA_BACKEND_BYTES", "value": "4294967296"},
                        {"name": "ETCD_SNAPSHOT_COUNT", "value": "50000"},
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "etcd-data",
                    "emptyDir": {},
                }
            ],
        },
    }



