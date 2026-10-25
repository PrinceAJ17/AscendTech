# AI prompts

**Tool used:** Claude (Anthropic), model Claude Opus 5.5, via Claude Code in the Claude desktop app. It was used as a reviewer, tutor and pair programmer: critiquing my initial approach, explaining concepts, writing and commenting code, and analysing my test recordings and screen captures.

Below are all the prompts I sent, in order, in one continuous session. Notes in *[brackets]* add context:
- *[reply button]* marks a reply I sent by clicking one of the assistant's suggested-reply buttons rather than typing it.
- *[quoting: …]* marks a reply to a specific passage of the assistant's previous answer.
- *[multiple choice]* marks a choice I picked when the assistant asked a question with set options.

## 1. Reviewing my approach

1. I was recently shortlisted for an interview and I received a take home assignment as a part of it. I have written down my own thought process and justification on which approach would be the best way to go about the whole project but I need you to:
   1) Explain to me the current faults and positives with my approach and whether I have understood the takeaway from the question.
   2) Validate where I was right and point out where I was wrong when it came to the technicalities of the technologies I mentioned.

2. Only rewrite it by first acknowledging the mistakes that I made very briefly. Then I want you to write up a plan to approach Option A while explaining the choices we make.

3. I need some clarification on the points you have presented in the plan:
   1) Explain OpenPnP to a beginner
   2) Expand on the no server round trip point. Do you mean that processing would be quicker when we do not have some sort of video processing that involves sending frames to some sort of server?
   3) Since we will be using MediaPipe for most of this project, please explain what exactly MediaPipe is and how it entails our project.
   4) Explain to me the math involved in how glasses would physically fit on a face.
   5) Explain why we are using a One Euro filter and not something like kalmans filter
   6) I will need you to explain the workings of Three.js here and how it will be used.
   7) I need you to explain how performance is measured here a bit more clearly
   8) For testing limitations I need you to explain what sort of stress test measurements you will be implementing
   9) I would also need you to explain how we would be building a system here that should not depend too heavily on a persons's computer specs

   Please explain all these questions in detail

## 2. Prototype UI, webcam and face landmarks (steps 1–2)

4. Let us start with the implementation of this project, can we try to first make a very basic UI for testing/prototyping purposes then implement a webcam frame?
5. What is next and how do we implement it
6. Yes, let us implement step 1 but while implementing it I want you to make code comments in the codebase to allow me to understand what everything does
7. Do we start measuring FPS at this stage?
8. Explain the next step and let us move on to it while adhering to what I said about code comments

## 3. 3D glasses and occluder tuning (step 3)

9. The 2D glasses fit well in Chrome. Let's move to step 3, the 3D version with Three.js. *[reply button]*
10. *[quoting: the explanation of building the head pose from landmarks instead of MediaPipe's matrix]* Could you just make this a point in the thoughtprocess md file

    The issue right now is that the occluder is not directly on my face but rather its a few positions up. It only aligns well with my face when I tilt my head up. Is it supposed to be on purpose?
11. I used my actual glasses to line it up properly and right now the pitch correction should be correct by -2 degrees and the height should be adjusted by -12mm . Would using my actual glasses be a good way to measure or would it be considering as an overfitting if it is adjusted to my shape measurements only?

## 4. Smoothing (step 4)

12. What is the next step
13. Yes implement it and let me see where we will have to adjust it
14. What I noticed was that there was not much jitter but when I turned smoothing off it felt still but the glasses did not "drag" across my face when I moved. When smoothing is on the glasses slightly "dragged across my face. The hertz value was 1 while the beta was 30
15. I think 3Hz is good enough where it might have a very slight jitter but the glasses do not drag

## 5. Lighting (step 5)

16. I've commited let us move onto step 5
17. It looks natural but the smoothing feels like it is doing a bit more damage than good right now because even when I keep the beta as 50 it still has a drag to it. I don't want you to change anything much right now but rather just keep it as a note in thought processes
18. What do you think about this issue is this normal or something that we can ignore for the time being

## 6. Measurement and stress tests (step 7)

19. Let us move onto the next step
20. Should I untick the fixed lighting label anytime while doing the test
21. And I am assuming we do not change the beta or hertz min cutoff values as well?
22. I have finished, could you check my downloads folder to see the recorded csv files
23. Could you first note down these findings in the thought process md file
24. *[quoting: "Our work per frame is about 12 ms of inference plus 0.2–2.6 ms of rendering. That's enough capacity for roughly 60 FPS."]* Could you explain what this means
25. *[quoting: the finding about render spikes from lighting estimation, and the suggestion to record a run with Match room lighting off]* I ran this experiment could you analyze how it faired
26. Yes, add the lighting on/off comparison to finding 5 in ThoughtProcess.md. *[reply button]*

## 7. Camera frame rate investigation

27. Apply the fixes: smoothing off by default and request 30 FPS from the camera
28. I have recorded both for bright and dim light and it is in the downloads folder
29. Yes, log the camera's capabilities (frame rate range and exposure modes) so we can see whether we can push it to 30 FPS. *[reply button]*
30. *[quoting: "Camera capabilities under Debug"]* where do I find this, in dev tools?
31. *[pasted the JSON from the Camera capabilities panel: ASUS FHD webcam, frame rate up to 30, manual and continuous exposure modes, exposure time 39–5000, brightness −64 to 64]*
32. Could you first explain what manual exposure means and what exactly we are trying to do
33. Could you add the limitations we are facing right now into the thought process md file before we move on
34. Yes, add the manual exposure control (checkbox, exposure slider up to 31.25 ms, brightness slider) and log the settings in the CSV. *[reply button]*
35. I think there is something wrong with the conditions you added "…\Videos\Captures\Glasses Try-On - Google Chrome 2026-10-25 01-42-30.mp4" I have recorded the video of the setup when I trialed it. One of the issues is that the camera just doesn't show a natural input even after a refresh of the page and the values being changed does not show a proper visual. Analyse the video carefully to diagnose what went wrong with the changes
36. "…\Videos\Captures\Glasses Try-On - Google Chrome 2026-10-25 02-04-56.mp4" the manual exposure still seems very skewed I do not think this is how it is supposed to look like. The exposure level and brightness levels do not make it any better to see the face
37. Yes, remove the manual exposure and brightness controls (keep the startup reset and the capabilities panel), and note the result in ThoughtProcess.md. *[reply button]*
38. *[pasted the `resetCameraSettings` function]* could I comment out this function then?
39. I think I will keep it in the case of edge cases but note this down

## 8. Submission (step 8)

40. Let us move onto step 8

## 9. UI redesign

41. *[run through the `/apple-design` skill command]* I want to further refine this. I want to make the ui much more interactive and better to look at. Use the ui-ux-pro-max-skill as well as other UI/UX design skills that I have installed for you
42. *[multiple choice]* Layout: **Full-screen camera** (camera fills the window, floating frosted-glass controls, frame picker tray at the bottom, slide-out drawer for developer tools). Developer tools: **Hidden by default** (a toggle reveals debug, stats and measurement).
43. The new UI looks good in Chrome. Let's go back to deploying the live demo. *[reply button]*

## 10. Deployment

44. *[multiple choice]* Host for the live demo: **Vercel**.
45. Let me make the github public then just for the time being
46. Its public, let us move onto deployment
47. ive committed it could you update the agentprompts.md file
