using StaticArrays

const TRIANGLE_CASES::Array{Int32, 2} = [-1 -1 -1 -1 -1 -1 -1;
3 0 2 -1 -1 -1 -1;
1 0 4 -1 -1 -1 -1;
2 3 4 2 4 1 -1;
2 1 5 -1 -1 -1 -1;
5 3 1 1 3 0 -1;
2 0 5 5 0 4 -1;
5 3 4 -1 -1 -1 -1;
4 3 5 -1 -1 -1 -1;
4 0 5 5 0 2 -1;
5 0 3 1 0 5 -1;
2 5 1 -1 -1 -1 -1;
4 3 1 1 3 2 -1;
4 0 1 -1 -1 -1 -1;
2 0 3 -1 -1 -1 -1;
-1 -1 -1 -1 -1 -1 -1] .+ 1

const EDGES::Array{Int32, 2} = [
0 1 ;
1 2 ;
2 0 ;
0 3 ;
1 3 ;
2 3 ] .+ 1

@inline function GetPoint!(pts, idx, pt)
  for i in 1:3
    pt[i] = pts[3*(idx-1)+i]
  end
  nothing
end

function ComputeIndex(scalars, isovalue)::Int32
  CASE_MASK = SVector{4, Int32}(1, 2, 4, 8)

  index::Int32 = 0
  for i in 1:4
    if scalars[i] >= isovalue
      index |= CASE_MASK[i]
    end
  end
  return index + 1
end

@inline function GetCellPoints!(conn, pts, cellPts)
  apt::MVector{3, Float64} = MVector{3, Float64}(undef)
  nverts = length(conn)
  for i in 1:nverts
    GetPoint!(pts, conn[i], apt)
    for j in 1:3
      @inbounds cellPts[3*(i-1)+j] = apt[j]
    end
  end
  return nothing
end

@inline function GetCell!(conn, offs, idx, cellConn)
  for i in 1:4
    cellConn[i] = conn[offs[idx]+i-1]
  end
  nothing
end

@inline function GetCellValues!(conn, var, values)
  nverts = length(conn)
  for i in 1:nverts
    values[i] = var[conn[i]]
  end
  return nothing
end

using Metal

function CountTriangles(contourValue, ncells, conn::MtlDeviceArray, offs::MtlDeviceArray, data::MtlDeviceArray, ntrisout::MtlDeviceArray, triangle_cases::MtlDeviceArray)
  index = thread_position_in_grid_1d()
  stride = threads_per_threadgroup_1d()
  cellConn::MVector{4, Int32} = MVector{4, Int32}(undef)
  cellValues::MVector{4, Float32} = MVector{4, Float32}(undef)
  for cellIdx in index:stride:ncells
    GetCell!(conn, offs, cellIdx, cellConn)
    GetCellValues!(cellConn, data, cellValues)
    ntris::UInt32 = 0
    idx = 1
    while triangle_cases[ComputeIndex(cellValues, contourValue), idx] > 0
      ntris += 1
      idx += 3
    end
    ntrisout[cellIdx] = ntris
  end
  return
end

function ContourCells(contourValue, ncells, conn::MtlDeviceArray, offs::MtlDeviceArray, pts::MtlDeviceArray, data::MtlDeviceArray, ntris::MtlDeviceArray, triangle_cases::MtlDeviceArray, edges::MtlDeviceArray)

  cellConn::MVector{4, Int32} = MVector{4, Int32}(undef)
  cellValues::MVector{4, Float32} = MVector{4, Float32}(undef)

  cellIdx = 1
  if ntris[cellIdx] < 1
    return
  end
  GetCell!(conn, offs, cellIdx, cellConn)
  GetCellValues!(cellConn, data, cellValues)

  idx = 1
  itri = 0
  tidx = ComputeIndex(cellValues, contourValue)
  while triangle_cases[tidx, idx] > 0
    for i in idx:idx+2
      eidx = triangle_cases[tidx, i]
      vs1 = edges[eidx, 1]
      vs2 = edges[eidx, 2]
      deltaScalar::Float32 = cellValues[vs2] - cellValues[vs1]
    end
    idx += 3
  end
  
  return
end

using HDF5

function Contour(contourValue)
  fid = h5open("tet-mid.h5", "r")  
  offs_::Vector{Int32} = read(fid["offsets"])
  offs = MtlVector(offs_)
  pts_::Vector{Float32} = read(fid["points"])
  pts = MtlVector(pts_)
  data_::Vector{Float32} = read(fid["data"])
  data = MtlVector(data_)
  conn_::Vector{Int32} = read(fid["connectivity"])
  conn = MtlVector(conn_)
  ncells::Int32 = length(conn_)/4
  ntrisGPU = MtlVector{Int32}(undef, ncells; storage=Shared)
  ntris = unsafe_wrap(Array{Int32}, ntrisGPU, size(ntrisGPU))
  triangle_cases = MtlArray(TRIANGLE_CASES)
  edges = MtlArray(EDGES)

  @time begin
  Metal.@sync @metal threads=512 groups=1 CountTriangles(contourValue, ncells, conn, offs, data, ntrisGPU, triangle_cases)
  synchronize()
  end
  println(ntris)

  triOffsets::Vector{Int32} = Array{Int32, 1}(undef, ncells+1)
  triOffsets[1] = 0
  triOffsetsp = @view triOffsets[2:end]
  accumulate!(+, triOffsetsp, ntris)
  triOffsetsGPU = MtlVector{Int32}(triOffsets)

  nTotTris = triOffsets[end]
  tris = Array{Float64, 1}(undef, nTotTris*3*3)

  @time begin
  Metal.@sync @metal threads=1 groups=1 ContourCells(contourValue, ncells, conn, offs, pts, data, ntrisGPU, triangle_cases, edges)
  synchronize()
  end
    
  return
end

Contour(130)
