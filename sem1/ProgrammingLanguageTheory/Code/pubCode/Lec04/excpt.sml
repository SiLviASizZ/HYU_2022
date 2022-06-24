exception MyUndesirableCondition

exception MyOtherException of int * int

fun mydiv (x,y) =
    if y=0
    then raise MyUndesirableCondition
    else x div y 

exception MyException of int * string
fun raiseExcpt(x) =
    if x div 2 = 0
    then raise MyException(1, "even")
    else raise MyException(2, "odd")



fun good_max (xs : int list) = 
    if null xs
      then 0 (* horrible style; fix later *)
    else if null (tl xs)
      then hd xs
    else 
        let val max_rest = good_max(tl xs)
        in
            if hd xs > max_rest
            then hd xs
            else max_rest
        end
 
(* maxlist: int list * exception -> int *)
fun maxlist (xs: int list) =
  case xs of


(*
fun maxlist (xs, ex) =
  case xs of
*)
