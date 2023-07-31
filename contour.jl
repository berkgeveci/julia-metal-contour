using StaticArrays
using HDF5
using Metal
using KernelAbstractions
using Adapt

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

struct UnstructuredGrid{T}
  connectivity :: T
  offsets :: T
  ncells :: Int32
end

function Adapt.adapt_structure(to, from::UnstructuredGrid)
  connectivity = adapt(to, from.connectivity)
  offsets = adapt(to, from.offsets)
  UnstructuredGrid(connectivity, offsets, from.ncells)
end

#function Adapt.adapt_structure(to, from::UnstructuredGrid)
#  connectivity = adapt(to, from.connectivity)
#  offsets = adapt(to, from.offsets)
#  UnstructuredGrid(connectivity, offsets, from.ncells)
#end

struct Cell
  grid
  idx
end

struct CellValues
  cell
  data
end

@inline function ComputeIndex(scalars, isovalue)::Int32
  CASE_MASK = SVector{4, Int32}(1, 2, 4, 8)

  index::Int32 = 0
  for i in 1:4
    if scalars[i] >= isovalue
      index |= CASE_MASK[i]
    end
  end
  return index + 1
end

@inline function GetCell(grid, idx)
  return Cell(grid, idx)
end

@inline function GetCellValues(cell, var)
  return CellValues(cell, var)
end

import Base.getindex

@inline function Base.getindex(c::Cell, i)
  @inbounds c.grid.connectivity[c.grid.offsets[c.idx]+i-1]
end

@inline function Base.getindex(c::CellValues, i)
  @inbounds c.data[c.cell[i]]
end

@kernel function CountTriangles(ntrisOut, contourValue, grid, data, triangle_cases)
  index = @index(Local)
  stride = Int32(@groupsize()[1])

  for cellIdx in index:stride:grid.ncells
    cell = GetCell(grid, cellIdx)
    cellValues = GetCellValues(cell, data)
    ntris::UInt32 = 0
    idx = 1
    while triangle_cases[ComputeIndex(cellValues, contourValue), idx] > 0
      ntris += 1
      idx += 3
    end
  ntrisOut[cellIdx] = ntris
  end
end

# function ContourCells(contourValue, ncells, conn::MtlDeviceArray, offs::MtlDeviceArray, pts::MtlDeviceArray, data::MtlDeviceArray, ntris::MtlDeviceArray, triangle_cases::MtlDeviceArray, edges::MtlDeviceArray)

#   cellConn::MVector{4, Int32} = MVector{4, Int32}(undef)
#   cellValues::MVector{4, Float32} = MVector{4, Float32}(undef)

#   cellIdx = 1
#   if ntris[cellIdx] < 1
#     return
#   end
#   GetCell!(conn, offs, cellIdx, cellConn)
#   GetCellValues!(cellConn, data, cellValues)

#   idx = 1
#   itri = 0
#   tidx = ComputeIndex(cellValues, contourValue)
#   while triangle_cases[tidx, idx] > 0
#     for i in idx:idx+2
#       eidx = triangle_cases[tidx, i]
#       vs1 = edges[eidx, 1]
#       vs2 = edges[eidx, 2]
#       @inbounds deltaScalar::Float32 = cellValues[vs2] - cellValues[vs1]
#   end
#     idx += 3
#   end
  
#   return
# end

function ReadGrid(fname::AbstractString)
  fid = h5open(fname, "r")  
  offs_::Vector{Int32} = read(fid["offsets"])
#  offs = adapt(backend, offs_)
  pts_::Vector{Float32} = read(fid["points"])
  pts = adapt(backend, pts_)
  data_::Vector{Float32} = read(fid["data"])
  data = adapt(backend, data_)
  conn_::Vector{Int32} = read(fid["connectivity"])
#  conn = adapt(backend, conn_)
  ncells::Int32 = length(conn_)/4

  UnstructuredGrid(conn_, offs_, ncells), data
end

function Contour(backend, grid, data, contourValue)

  triangle_cases = adapt(backend, TRIANGLE_CASES)
  edges = adapt(backend, EDGES)

  ntrisOut = KernelAbstractions.allocate(backend, Int32, Int64(grid.ncells))
  CountTriangles(backend, 16)(ntrisOut, contourValue, grid, data, triangle_cases, ndrange=64)

  println(ntrisOut)

  # triOffsets::Vector{Int32} = Array{Int32, 1}(undef, ncells+1)
  # triOffsets[1] = 0
  # triOffsetsp = @view triOffsets[2:end]
  # accumulate!(+, triOffsetsp, ntris)
  # triOffsetsGPU = MtlVector{Int32}(triOffsets)

  # nTotTris = triOffsets[end]
  # tris = Array{Float64, 1}(undef, nTotTris*3*3)

  # @time begin
  # Metal.@sync @metal threads=1 groups=1 ContourCells(contourValue, ncells, conn, offs, pts, data, ntrisGPU, triangle_cases, edges)
  # synchronize()
  # end
    
  return
end

backend = MetalBackend()
grid, data = ReadGrid("tet-mid.h5")
Contour(backend, grid, data, 130)
#Contour(MetalBackend(), 130)
