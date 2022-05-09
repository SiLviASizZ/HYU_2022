(* returns true if for all element e in alist
 *    test(e) = true
 *)
fun ifall (test, alist) = 
  case alist of
       [] => false
     | hd::[] => test(hd)
     | hd::rest => test(hd) andalso ifall(rest)

(* returns true if for any element e in alist
 *    test(e) = true
 *)
fun ifany (test, alist) = 
  case alist of
       [] => false
     | hd::[] => test(hd)
     | hd::rest => test(hd) orelse ifall(rest)

(* returns a list of elements in xs that satisfy 
 *    f(e) = true
 *   e.g. f = fun positive(x) = x > 0 
 *        xs = [~1, 0, 5, 1, 22, ~3]
 *        myfilter(f, xs) => [5, 1, 22]
 *)
fun myfilter (f, xs) =
  case xs of
    [] => []
  | x::xs' => if f(x) then x::myfilter(f, xs') 
              else myfilter(f, xs') 

