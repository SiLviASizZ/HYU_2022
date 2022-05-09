(* Type Inference *)

(* simulate type checker (in a way) *)

(* this is given:  abs: int -> int *)

fun f x = 
   let val (y,z) = x in
       (abs y) + z
   end
(*
 f: T1 -> T2
 x: T1

 T1 = T3*T4
 y: T3
 z: T4
 
 T3 = int
 T4 = int
 T2 = int

 f: int*int -> int
 x: int*int
*)

fun sum xs =
   case xs of
     [] => 0
   | x::xs' => x + (sum xs')
(*
 sum: T1 -> T2
 xs: T1

 T1 = T3 list
 T2 = int

 xs' = T1 = T3 list
 x = T3
 
 T3 = int

 sum: int list -> int
 xs: int list
*)

fun broken_sum xs =
    case xs of
	    [] => 0
    | x::xs' => x + (broken_sum x)
(*
   broken_sum: T1 -> T2
   xs: T1

   T1 = T3 list
   T2 = int
   
   xs': T3 list
   x: T3
   
   T1 = T3 = T3 list #ERROR
 *)

fun length xs =
    case xs of
     [] => 0
   | x::xs' => 1 + (length xs')
(*
  length: T1 -> T2
  xs: T1

  T1 = T3 list
  T2 = int

  xs': T3 list
  x: T3
 
  length: 'a list -> int
*)

fun f (x,y,z) =
    if true
    then (x,y,z)
    else (y,x,z)
(*
   f: T1 * T2 * T3 -> T4
   x: T1, 
   y: T2,
   z: T3,

   T1 = T2
   T4 = T1*T1*T3

   f: 'a*'a*'b-> 'a *'a*'b
*)

fun compose (f,g) = fn x => f (g x)

(* compose: T1 * T2 -> T3
   f: T1
   g: T2 
   anon func: T3 

   x: T4,
   T2 = T4 -> T5
   T1 = T5 -> T6

   T3 = T4 -> T6 

   compose: (T5->T6)*(T4->T5) -> (T4->T6)
             'a->'b * 'c->'a  ->  'c->'b
*)
