[shader("vertex")]
float4 shared(uint vertex_id : SV_VertexID) : SV_Position
{
    return float4(vertex_id, 0, 0, 1);
}

[shader("fragment")]
float4 shared(float4 position : SV_Position) : SV_Target
{
    return position;
}
