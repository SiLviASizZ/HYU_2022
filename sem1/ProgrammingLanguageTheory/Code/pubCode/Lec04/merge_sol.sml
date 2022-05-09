fun merge1 (a:int list, b:int list) = 
  if null a then b
  else if null b then a
  else if hd a < hd b then 
    hd(a)::merge1(tl(a), b)
  else 
    hd(b)::merge1(a, tl(b))

fun merge (a:int list, b:int list) = 
let fun merge_aux(x, y, ans) =
      if null x then y@ans
      else if null y then x@ans
      else if hd x > hd y then 
        merge_aux(tl(x), y, (hd x)::ans)
      else 
        merge_aux(x, tl(y), (hd y)::ans)
in 
    merge_aux(rev(a), rev(b), [])
end

fun merge2 (a:int list, b:int list) = 
let fun merge_aux(x, y, ans) =
      if null x then rev(ans)@y
      else if null y then rev(ans)@x
      else if hd x < hd y then 
        merge_aux(tl(x), y, (hd x)::ans)
      else 
        merge_aux(x, tl(y), (hd y)::ans)
in 
    merge_aux(a, b, [])
end


val l1 = [1,5,8,11,14,20]
val l2 = [~10,0,5,9,100]


