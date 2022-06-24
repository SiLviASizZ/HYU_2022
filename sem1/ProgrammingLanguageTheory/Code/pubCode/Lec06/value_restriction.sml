(* a function that takes a list and returns a list of pairs (paired to 1)
 * e.g. ["monkey", "pirate", "island"] 
 *       => [ ("monkey", 1), ("pirate", 1), ("island", 1) ]
 *)


val pairWithOne = List.map (fn x => (x,1)) 


(* workarounds: *)
fun pairWithOne xs = 
    List.map (fn x => (x,1)) xs

val pairWithOne : string list -> (string * int) list =
    List.map (fn x => (x,1))
