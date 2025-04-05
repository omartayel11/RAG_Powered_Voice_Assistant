from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://58ffcb38-a543-4f91-81c1-358d4af0bf83.eu-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.E_d_nSkUDPbwKg4-cyqoygB9oK8auQbnn77P9BBXglQ",
)

print(qdrant_client.get_collections())