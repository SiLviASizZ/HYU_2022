(*
signature MATHLIB =
sig
val fact : int -> int
val half_pi : real
val doubler : int -> int 
val doubler : string -> string
end

*)


structure MyMathLib =
struct

fun fact x =
    if x=0
    then 1
    else x * fact (x - 1)

(*val half_pi = Math.pi / 2.0*)

fun doubler y = y + y

val xyz = doubler 42;

end

(*
val pi = MyMathLib.half_pi + MyMathLib.half_pi
val err = MyMathLib.doubler 42
*)
(* val twenty_eight = MyMathLib.doubler 14 *)

