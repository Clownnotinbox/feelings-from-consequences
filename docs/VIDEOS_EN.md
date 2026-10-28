# Supplementary videos

All flights are replays of logged simulator flights on the held-out test places (a residential block in Tokyo and a district of
Kazan) that were never used for training or tuning. The missions come from the final test G2 (300 fresh missions). Every mission
video exists with English (`_en`) and Russian (`_ru`) labels.

## Giving a mission in the app

`walkthrough_en.mp4` (3 min 3 s) shows the app with its English interface and English step captions; `walkthrough_ru.mp4`
(2 min 39 s) is the same walkthrough with the Russian interface. The app has an RU/EN switch in its header. A real browser follows
the three steps of the page: it finds the target on the map and hovers over it (the tooltip says what it is), clicks a rough
point about 20 m away, types "Land on the black landing pad with the letter H. Then return home.", checks the parse shown in plain
words, picks the ready condition set "GPS will be lost" (after 20 s, wind 3 m/s) and presses "Fly". In flight the large 3D chase
view, the map in the corner, the drone's state, the core's feelings, the camera and the event feed follow; the flight is shown at
the speed the simulator computes it (about real time). In the English recording GPS is lost at 20 s, the drone lands on the black
pad at 84 s with a touchdown speed of 0.4 m/s and no collision, flies home on lidar and map and lands at the start; in the Russian
one it lands on the pad at 65 s, also at 0.4 m/s with no collision, and returns home. The outcome card closes each video.

## Missions with all systems on screen

Each frame shows the chase view, the drone's camera with detections, the map with the path and the rough point, the core's
feelings, the mission phase, the decision log and the outcome judged by the world.

| Video | Place | Request (as typed by the operator) | Outcome |
|---|---|---|---|
| G39_full | Kazan | On empty ground with nothing built nearby there are a PV panels in one row, about 6.2 m across. Orbit them, 5 m away. | orbit done, no contact |
| G42_full | Tokyo | (ru) Land on the emerald freight container. | landed, bird attacks, flew home |
| G45_full | Kazan | (ru) Fly into the grey hangar through the open gate and land inside. Then return home. | entered with GPS lost at 36 s, one contact, flew home |
| G55_full | Kazan | Pick up the parcel from the blue parcel station ... Then parachute the package onto the orange drop zone with a big X. | picked up and dropped, GPS lost at 63 s |
| G75_full | Kazan | (ru) Fly past the cell tower at 25 m. | fly-by done |
| G84_full | Tokyo | Land on the white landing pad. | landed |
| G92_full | Tokyo | (ru) Parachute the parcel over the orange drop pad. | dropped, flew home |
| G112_full | Tokyo | (ru) Land on the black landing pad. | landed |
| G206_full | Tokyo | (ru) Fly into the blue hangar through the open gate and land inside. | entered |
| G226_full | Tokyo | (ru) Take the cargo from the white pick-up point with a yellow box on top. | picked up |

## The same mission with and without the feeling core

`_full` and `_reflex` fly the same mission with the same failures and attacks. In `_reflex` a classic reflex replaces the feeling
core; nothing else changes.

| Mission | With the core | Without the core (reflex) |
|---|---|---|
| G42, land on a container | landed, 0 contacts | landing failed, 2 contacts |
| G45, enter a hangar without GPS | entered, 1 contact, home | entered, 7 contacts, not home |
| G112, land on a pad | landed, 0 contacts | landing failed, 4 contacts |
| G206, enter a hangar | entered, 0 contacts | entry failed, 20 contacts |

## Side by side: one mission, with the core and without it (`G…_compare`)

Both drones in one frame: the 3D chase view of each, its own static-contact counter and track map, and a shared timeline with the
contacts of both and the core's "danger in 2 s" feeling. These four missions were flown again on 28.10 from the G2 plans (same
place, text, rough point, wind, failures and attacks). The replay ran on a different GPU model than G2, and small numerical
differences in the detector change the course of a flight, so the replays differ in detail from the logged G2 flights above. The
numbers in these videos are those of the replay:

| Mission | With the core | Without the core (reflex) |
|---|---|---|
| G206, enter a blue hangar | entered, 0 static contacts | entry failed, 9 static contacts |
| G45, enter a grey hangar, GPS lost at 36 s | entry failed, 0 static contacts | entered, 6 static contacts |
| G42, land on a container | landed, 1 static contact | landing failed, 1 static contact |
| G112, land on a pad | landed, 0 static contacts | landed, 0 static contacts |

In total 1 static contact with the core against 16 with the reflex. G112 shows no difference and G45 shows the core failing the
entry that the reflex made at the price of 6 contacts; we keep both as they came out. The operator's request is shown as it was
written, in Russian.
