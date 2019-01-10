#grover
#C = [-1/2 1/2 1/2 1/2;1/2 -1/2 1/2 1/2;1/2 1/2 -1/2 1/2;1/2 1/2 1/2 -1/2]
#hadamard
C = [1 1 1 1;1 -1 1 -1;1 1 -1 -1;1 -1 -1 1]/2
right_m = zeros(Float64,4,4)
right_m[1,:] = C[1,:]
left_m = zeros(Float64,4,4) 
left_m[2,:] = C[2,:]
up_m = zeros(Float64,4,4)
up_m[3,:]  = C[3,:]
down_m = zeros(Float64,4,4)
down_m[4,:]  = C[4,:]

using ProgressMeter
using LinearAlgebra
using Plots
gr()
function quantum_walk(loop)
    one_side = 50
    psi = zeros(Float64,one_side,one_side,4)
    psi[div(one_side,2),div(one_side,2),:]=[0.5,0.5,-0.5,-0.5]
    
    progress = Progress(loop)
    anim=@animate for t in 0:loop
        if t == 0
            continue
        else
            next_psi = zeros(Float64,one_side,one_side,4)
            for x in 1:one_side, y in 1:one_side
                x0 = ((x-1 + (one_side-1)) %one_side) + 1
                x1 = ((x+1 + (one_side-1)) %one_side) + 1
                y0 = ((y-1 + (one_side-1)) %one_side) + 1
                y1 = ((y+1 + (one_side-1)) %one_side) + 1
                next_psi[x,y,:] = copy( right_m*psi[x0,y,:] + left_m*psi[x1,y,:] + up_m*psi[x,y0,:] + down_m*psi[x,y1,:] )
            end
        end
        psi = copy(next_psi)
        next!(progress)
        
    hist = zeros(Float64,one_side,one_side)
    for x in 1:one_side, y in 1:one_side
        hist[x,y] = copy(dot(psi[x,y,:],psi[x,y,:]))
    heatmap(hist,aspect_ratio=1,cbar=true,cbar_lims=(0,0.008))
        
    end
    gif(anim,fps=20)
    

end
quantum_walk(200)
