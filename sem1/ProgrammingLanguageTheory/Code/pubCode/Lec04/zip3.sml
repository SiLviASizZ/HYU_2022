
exception ListLengthMismatch;

(* ([1,2,3,4], [10,20,30,40], [100,200,300,400])
   ==> [(1,10,100), (2, 20, 200), (3, 30, 300), (4, 40, 400)]
 *)
fun zip3 (lists) = 
    case lists of

(* [(1,10,100), (2, 20, 200), (3, 30, 300), (4, 40, 400)]
   ==> ([1,2,3,4], [10,20,30,40], [100,200,300,400])
 *)
fun unzip3 (triples) = 
  case triples of
