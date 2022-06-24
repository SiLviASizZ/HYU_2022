(* f is a normal function
   f(x,y)
*)
fun add(x,y) = x+y

fun curry f x y = f(x,y)

fun curry f = fn x => fn y => f(x,y)

val curried_add2 = curry add
(* f is a curried function
   f x y 
*)

fun curried_add x y = x+y


fun uncurry f (x,y) = f x y


(* change the arguments order 
 * f x y => f y x *)
fun other_curry1 f x y = f y x

fun other_curry2 f = 
    fn x => fn y => f y x

(* example *)
(* tupled but we wish it were curried *)
(* range(1, 3) ==> [1,2] *)
fun range (i,j) =
    if i > j
    then [] 
    else i :: range(i+1, j)
(* function counting from 1 *)
val count_upto = ((curry range) 1)

(* xs = [1, 2, ..., 7] *)
val xs = count_upto 7

(* count10from 7 ==> [7, 8, 9, 10] *)
val count10from = other_curry2 (curry range) 10

val sevenToTen = count10from 7
