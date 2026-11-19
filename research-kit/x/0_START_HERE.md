# X (Twitter) research kit for the "nfl" project

This folder lets you collect what people argue about on X from your own laptop and
hand it back to the "nfl" chat, which cannot reach X itself.

## Steps

1. On your laptop, open Chrome and log in to x.com as usual.
2. Open Claude in Chrome (the Claude browser extension). If you do not have it, the
   Claude desktop app with browser access works the same way.
3. Open `1_PROMPT_paste_this_into_Claude.md` from this folder, copy all of it, and
   paste it into Claude as your message.
4. Let it work. It only reads public posts. It does not post, like, follow or message.
   If X shows a login, captcha or rate-limit screen, it stops and tells you.
5. When it finishes, copy its whole answer into `2_twitter_findings.md` in this folder
   (replace the placeholder text).
6. Optional: put YouTube links or titles in `3_videos.md`, one per line, with a
   sentence on what each video argues if you like.
7. Send the results back by pushing this folder to GitHub:

   ```bash
   git add research-kit/x
   git commit -m "Add X findings and video list"
   git push
   ```

   The "nfl" chat reads `research-kit/x/2_twitter_findings.md` and
   `research-kit/x/3_videos.md` from the branch. Pasting the contents into the chat
   also works.

## Tips

- If the answer is cut off, tell Claude "continue" and paste both parts into the file.
- If you want it narrower or faster, tell it to do only search group 1 (the Lions)
  and section 5 (questions nobody answers well).
- Nothing in this folder contains passwords or account details, and none are needed.
