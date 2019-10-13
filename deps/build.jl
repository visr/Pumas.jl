s = open("../LICENSE.md") do file
    read(file, String)
end

println(stdout,s)
println(stdout,"Note: Pumas is free for non-commercial use, but requires a license for commercial use. Please see the full license text above for more details.")
