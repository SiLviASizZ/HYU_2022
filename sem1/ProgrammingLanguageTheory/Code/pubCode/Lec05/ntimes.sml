fun double_n_times(n, v) =
    if n=0
    then v
    else double_n_times(n-1, v*2)

fun increment_n_times(n, v) =
    if n=0
    then v
    else increment_n_times(n-1, v+1)

fun nth_elm(n, lst) =
    if n=0
    then hd(lst)
    else nth_elm(n-1, tl lst)

(*
 * f(f(... f(x)  )) (apply f n times)
 *)
fun n_times (f, n, x) =


fun double x =
    x + x
fun increment x =
    x + 1

val x1 = n_times(double,4,3) 
val x2 = n_times(increment,4,7) 
val x3 = hd(n_times(tl, 2, [4,8,12,16])) 

fun double_n_times (n,x) = n_times(double,n,x)

fun nth_tail (n,x) = n_times(tl,n,x)
