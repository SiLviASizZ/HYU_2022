


(*
 given f and xs=[x1, x2, ..., xn]
 returns  [f(x1), f(x2), ..., f(xn)]
 *)
fun map (f, xs) =
  case xs of
      [] => []
    | x::xs' => f(x)::map(f, xs')
val intlist = [1, 2, 3, 4, 5]

map(fn x=> x mod 2, intlist)

val strlist = ["a", "bc", "def", "ghij"]

(*
 given f and xs=[x1, x2, ..., xn]
 returns  [xi, xj, ..] if f(xi)=true.
 *)
fun filter(f, xs) = 
  case xs of
    [] => [] 
  | x::xs' => if f(x) 
              then x::filter(f, xs')
              else filter(f, xs')
              
val intlist = [1, 2, 3, 4, 5]
filter(fn x=> (x mod 2)=0, intlist)

(*
(* returns true if for all element e in alist
 *    test(e) = true
 *)
fun ifall (test, alist) = 
    case alist of
        [] => true
      | a::[] => 
      | a::alist' => 

(* returns true if for any element e in alist
 *    test(e) = true
 *)
fun ifany (test, alist) = 
    case alist of
        [] => true
      | a::[] => 
      | a::alist' => 

*)
