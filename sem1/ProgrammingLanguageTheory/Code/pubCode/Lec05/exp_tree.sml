datatype exp = Constant of int
             | Negate   of exp
             | Add      of exp * exp
             | Multiply of exp * exp

val myexp1 = Add(Constant(42), Multiply(Constant(40), Negate(Constant(10))));
(* 42 + (40 * (- 10)) *)

val myexp2 = Add(Constant(41), Multiply(Constant(40), Negate(Constant(10))));
(* 41 + (40 * (- 10)) *)



(* exp -> bool,   true *)
fun all_even (e) =
   case e of 
        Constant i      => (i mod 2) = 0
      | Negate e2       => all_even(e2)
      | Add(e1,e2)      => all_even(e1) andalso all_even(e2)
      | Multiply(e1,e2) => all_even(e1) andalso all_even(e2)

fun all_odd (e) = 
   case e of 
        Constant i      => (i mod 2) = 1
      | Negate e2       => all_odd(e2)
      | Add(e1,e2)      => all_odd(e1) andalso all_odd(e2)
      | Multiply(e1,e2) => all_odd(e1) andalso all_odd(e2)

fun all (test, e) =
   case e of 
        Constant i      => test i 
      | Negate e2       => all(test, e2)
      | Add(e1,e2)      => all(test, e1) andalso all(test, e1) 
      | Multiply(e1,e2) => all(test, e1) andalso all(test, e1)

(* another version of all_even *)
fun all_even(e) = all(fn i => (i mod 2)=0, e)

fun all_odd(e) = all(fn i => (i mod 2)=1, e)

fun any (test, e) =
   case e of 
        Constant i      => 
      | Negate e2       => 
      | Add(e1,e2)      => 
      | Multiply(e1,e2) => 

fun any_even(e) =

