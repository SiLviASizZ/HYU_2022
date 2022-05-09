(* a=[1,3,5,9,10]
 * b=[~1,0,8,11]
 * merge1 => [~1, 0, 1, 3, 5, 8, 9, 10, 11]
 *)
fun merge1 (a:int list, b:int list) = 
  if null a then b
  else if null b then a
  else if hd a < hd b then 
    hd(a)::merge1(tl(a), b)
  else 
    hd(b)::merge1(a, tl(b))

(* rev(a)=[10, 9, 5, 3, 1] 
 * rev(b)=[11, 8, 0, ~1] 
 *)
 (* ans = [5,  8, 9]
  *)
fun merge (a:int list, b:int list) = 
let fun merge_aux(x, y, ans) =
        if null x then rev(y) @ ans
        else if null y then rev(x) @ ans
        else if hd x < hd y then merge_aux(x, tl y,(hd y)::ans)
        else (* hd x >= hd y *)
            merge_aux(tl x, y,  (hd x)::ans)
in 
    merge_aux(rev(a), rev(b), [])
end

(* a=[1,3,5,9,10]
 * b=[~1,0,8,11]
 *   [3, 1, 0, ~1]
 *)
fun merge2 (a:int list, b:int list) = 
let fun merge_aux(x, y, ans) =
        if null x then rev(ans) @ y
        else if null y then rev(ans) @ x 
        else if hd x < hd y then merge_aux(tl x, y, (hd x)::ans)
        else (* hd x >= hd y *)
            merge_aux(x, tl y, (hd y)::ans)
in 
    merge_aux(a, b, [])
end

val l1 = [1,5,8,11,14,20]
val l2 = [~10,0,5,9,100]


