(* nice example of nested pattern.  *)

(* nondecreasing: int list -> bool
 *
 * e.g.  [1,1,2, 3, 4, 5, 5] => true
 *       [1, 2, 1, 3, 5] => false
 *)
fun nondecreasing xs =
    case xs of


datatype sgn = P | N | Z (* positive, negative, and zero *)

(* multsign: int * int -> sgn
 *
 * e.g. 1, 1 => P
 *      ~1, 10 => N
 *      ~10, ~42 => P
 *)
 (* int * int -> sig *)
fun multsign (x1:int, x2:int) = 
let fun sign(x) = 
in
    case (sign(x1), sign(x2)) of
end
         
fun len(l: 'a list) =
  case l of
    [] => 0
   |_::rest => 1+len(rest)




