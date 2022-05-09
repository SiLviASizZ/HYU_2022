
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





exception InvalidArgument 

(* int list -> int *)
fun max2(xs:int list) = 
  case xs of
       [] => raise InvalidArgument
     | x::[] => x
     | x::xs' => let val max_rest = max2(xs') in
 								   if x < max_rest then max_rest else x
                 end
exception InvalidArgument 

fun max3(xs:int list, exc) = 
  case xs of
       [] => raise exc
     | x::[] => x
     | x::xs' => let val max_rest = max3(xs', exc) in
                   if x < max_rest then max_rest else x
                 end


