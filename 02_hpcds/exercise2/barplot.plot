set term png
set output "graph.png"
set boxwidth 0.3
set style fill solid
set logscale y
set nokey
set ylabel "seconds"

plot "data.dat" u 0:2:0:xtic(1) w boxes lc var

