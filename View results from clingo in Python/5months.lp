
employee(1..10).
maxHoursPerWeek(580).
minHoursPerWeek(310).

minEmpMorning(2). 
maxEmpMorning(4).
minEmpAfternoon(2). 
maxEmpAfternoon(3).
minEmpNight(1). 
maxEmpNight(2).

minNights(10). 
maxNights(30).
minDays(30). 
maxDays(40).

days(150).
day(1..150).

% Shifts(id, name, hours).
shift(1,"morning",7).
shift(2,"afternoon",7).
shift(3,"night",10).
shift(4,"nightoff",0).
shift(5,"rest",0).
shift(6,"holiday",0).


% Each employee only works one shift in a day.
1 <= {assign(E, S, D) : shift(S,_,_)} <= 1 :- day(D), employee(E).

% Each employee works from 100 to 42 hours per week.
:- employee(E), maxHoursPerWeek(MAX), #sum{H,D : assign(E,S,D), shift(S,_,H)} > MAX.
:- employee(E), minHoursPerWeek(MIN), #sum{H,D : assign(E,S,D), shift(S,_,H)} < MIN.

% Each employee cannot work twice in 24 hours.
:- employee(E), assign(E, S1, D), assign(E, S2, D+1), S2 < S1, S2 <= 3, S1 <= 3.

% After two consecutive nights there is one rest day.
:- not assign(E,4,D), assign(E,3,D-2), assign(E,3,D-1).
:- assign(E,4,D), not assign(E,3,D-2).
:- assign(E,4,D), not assign(E,3,D-1).

% At least 2 rest days each 14 days.
:- employee(E), day(D), days(DAYS), D <= DAYS-13, #count{D1:assign(E,5,D1), D1>=D, D1 < D+14} < 2.

% 30 days of holidays.
:- employee(E), #count{D:assign(E,6,D)} <= 30.

% Each morning the number of employee can range from 2 to 4.
:- day(D), #count{E:assign(E,1,D)} > K, maxEmpMorning(K).
:- day(D), #count{E:assign(E,1,D)} < K, minEmpMorning(K).

% Each afternoon the number of employee can range from 2 to 4.
:- day(D), #count{E:assign(E,2,D)} > K, maxEmpAfternoon(K).
:- day(D), #count{E:assign(E,2,D)} < K, minEmpAfternoon(K).

% Each night the number of employee can range from 1 to 3.
:- day(D), #count{E:assign(E,3,D)} > K, maxEmpNight(K).
:- day(D), #count{E:assign(E,3,D)} < K, minEmpNight(K).


%Balance requirements
% Fair distribution (morning, afternoon, night)
% ---> morning
:- employee(E), #count{D : assign(E,1,D)} > MAXDAYS, maxDays(MAXDAYS).
:- employee(E), #count{D : assign(E,1,D)} < MINDAYS, minDays(MINDAYS).
% ---> afternoon
:- employee(E), #count{D : assign(E,2,D)} > MAXDAYS, maxDays(MAXDAYS).
:- employee(E), #count{D : assign(E,2,D)} < MINDAYS, minDays(MINDAYS).
% ---> night
:- employee(E), #count{D : assign(E,3,D)} > MAXNIGHTS, maxNights(MAXNIGHTS).
:- employee(E), #count{D : assign(E,3,D)} < MINNIGHTS, minNights(MINNIGHTS).

% ---> morning
:~ employee(E), #count{D : assign(E,1,D)} = X, minDays(MIN), maxDays(MAX), X >= MIN, X <= MAX, Y=|X-78|. [Y@1,E]
% ---> afternoon
:~ employee(E), #count{D : assign(E,2,D)} = X, minDays(MIN), maxDays(MAX), X >= MIN, X <= MAX, Y=|X-78|. [Y@1,E]
% ---> night
:~ employee(E), #count{D : assign(E,3,D)} = X, minNights(MIN), maxNights(MAX), X >= MIN, X <= MAX, Y=|X-60|. [Y@1,E]


#show assign/3.
