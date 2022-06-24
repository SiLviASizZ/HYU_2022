fun map (f, xs) =
  case xs of
      [] => []
    | x::xs' => f(x)::map(f, xs')

val strlist = ["a", "bc", "def", "ghij"]

fun filter(f, xs) = 
  case xs of
    [] => [] 
  | x::xs' => if f x then x::filter(f, xs') 
                     else filter(f, xs')

(* returns true if for all element e in alist
 *    test(e) = true
 *)
fun ifall (test, alist) = 
    case alist of

(* returns true if for any element e in alist
 *    test(e) = true
 *)
fun ifany (test, alist) = 
    case alist of

