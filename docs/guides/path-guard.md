# Path guard (`orkexa.fs`)

`orkexa.fs.SafeRoot` lets an agent's tools read files below one folder that a person granted, and nothing else.
It is product-neutral: Nodestro builds its folder grants and file tools on it (plan N-10), and any other host
can do the same. It uses only the standard library (`ctypes` on Windows and for Linux `openat2`).

```python
from orkexa.fs import FsError, SafeRoot, TextBudget

root = SafeRoot(
    "/home/ada/Documents/Offer 2026",
    protected=[product_data_dir],  # never a root, never reachable below one
    exclude=["**/*.key", "**/node_modules"],
    max_file_bytes=20 * 1024 * 1024,
)
grant_identity = root.identity  # store it with the grant

root.list("")  # [Entry(name, path, kind, size, ambiguous)]
root.read_bytes("notes/plan.md")
root.extract_text("specs/offer.pdf", budget=TextBudget(timeout=20, max_pages=200))

try:
    root.read_bytes("../.ssh/id_rsa")
except FsError as exc:
    exc.code  # "fs.parent_reference": map codes, never messages
```

Later, `SafeRoot(path, expect_identity=grant_identity)` refuses a folder that was swapped in under the same name
(`fs.root_changed`). `root.revoke()` refuses every later call and stops a read in progress at its next 1 MiB
chunk.

## How a path is checked

1. **Names first, the same on every system.** A path is relative, `/`-separated and normalized to NFC. Refused
   before anything is opened:
    * `..` (`fs.parent_reference`); absolute paths, drives (`C:`), UNC and device prefixes, a leading `~`
      (`fs.absolute_path`);
    * `\`, `:` (alternate data streams such as `file.txt:secret`), NUL and control characters, empty and `.`
      components, reserved device names (`CON`, `NUL.txt`, `COM1`, `LPT¹`), a trailing dot or space, 8.3 short
      names (`PROGRA~1`, `SECRET~1.TXT`), look-alike separators (fullwidth solidus, division slash, fraction
      slash), invalid UTF-8 (overlong encodings), more than 255 bytes per component or 4096 in all
      (`fs.invalid_path`).

    Nothing is decoded: `%2e%2e%2f` is a file name, not `../`. The Windows rules hold on Linux and macOS too, so a
    path never means a different file, or a device, on another system.
2. **Blocklist and patterns.** `blocked` (default `DEFAULT_BLOCKED`: `.ssh`, `.gnupg`, `.aws`, `.kube`,
   `.netrc`, `Library/Keychains`, the Windows credential folders and more) raises `fs.blocked`; `exclude` and
   `include` raise `fs.excluded`. Patterns use `*`, `?` and `**`, are anchored at the root unless they start
   with `**/`, and compare NFC plus case-folded on every system, so `.SSH`, `LIBRARY/KEYCHAINS`, `.ſsh` and the
   NFD spelling of a name are blocked as well. A pattern that matches a folder covers everything below it.
3. **The root, again.** Every call reopens the root and checks that it is the folder that was granted: the
   same identity (device and inode; volume serial number and file index on Windows) and, on macOS and Windows,
   the same final path. A root renamed and replaced by a link or another folder raises `fs.root_changed`.
4. **One component at a time, through descriptors.** No path string is resolved twice; the file is read from
   the very descriptor that passed the checks.

| System | How a component is opened | What is refused |
|---|---|---|
| Linux | `openat2(dirfd, name, RESOLVE_BENEATH \| RESOLVE_NO_SYMLINKS \| RESOLVE_NO_MAGICLINKS \| RESOLVE_NO_XDEV)` on kernels 5.6+, otherwise `openat(dirfd, name, O_NOFOLLOW)`; folders with `O_DIRECTORY`, files with `O_NONBLOCK \| O_NOCTTY` | symbolic links anywhere on the way (`fs.link`), magic links (`/proc/self/fd/N`), bind mounts and other file systems (`fs.cross_device`) |
| macOS | `openat(dirfd, name, O_NOFOLLOW)`, then `fcntl(F_GETPATH)` of every descriptor must lie below the root's (case-folded) | links, other volumes (`st_dev`), a path outside the root (`fs.outside_root`) |
| Windows | `CreateFileW("\\?\<root>\<a>\<b>", FILE_FLAG_OPEN_REPARSE_POINT \| FILE_FLAG_BACKUP_SEMANTICS)` per component, share mode without `FILE_SHARE_DELETE`; `GetFinalPathNameByHandleW` of every handle must be the root plus exactly the names asked for | symbolic links, junctions, volume mount points and other reparse points (`fs.link`), cloud placeholders (`fs.cloud_file`), a final path elsewhere (`fs.outside_root`) |

On Windows there is no documented `openat`, so the walk pins the path instead: a folder whose handle is open
without `FILE_SHARE_DELETE` cannot be renamed, deleted or replaced, so the next component resolves through
exactly the folders already checked. `FILE_FLAG_OPEN_REPARSE_POINT` alone protects only the last component (the
W0-6 spike showed that a folder link before it is followed), which is why every component is opened and checked
on its own, and why the final path of each handle is the binding check. `\\?\` turns off Win32 name rewriting
(trailing dots, device names); `FILE_NAME_NORMALIZED` expands 8.3 names. Nothing needs administrator rights.

Links are never followed, also links that stay inside the root; a listing shows them as `kind="link"`.

## Files that are refused

* **Special files**: FIFOs, sockets and devices (`fs.not_regular_file`), without blocking on a FIFO.
* **Hard links** (decision A-15): a file with `st_nlink > 1` (Windows `nNumberOfLinks`) is refused with
  `fs.hard_link`, because it may be the very file of another folder and no path shows that. `allow_hard_links=True`
  opts in per root (for example a backup folder). Folders are not affected.
* **Size**: a file above `max_file_bytes` (default 50 MiB) is refused from its size alone, before a byte is read
  (`fs.too_large`, "The file has N bytes; at most M bytes can be read"); a file that grows while it is read is
  cut off at the limit and refused too.

## Unicode and case

Input paths are normalized to NFC. A name stored in another form (NFD, as macOS and some tools write it) is found
when exactly one entry of the folder has that NFC form; when several have it, the exact name opens and any other
spelling raises `fs.ambiguous_name`. Listings mark such entries `ambiguous=True`. The blocklist, the patterns and
the protected-folder names compare NFC plus case-folded, which fails closed on case-insensitive file systems
(APFS, NTFS); protected folders are also compared by identity, so no spelling reaches them. Bidirectional
controls and zero-width characters stay readable (P0-2 decision) and `Entry.display` shows them escaped
(`invoice\u202efdp.exe`).

## Roots that cannot be granted

`SafeRoot` refuses, with `fs.protected_root`:

* a file-system or drive root (`/`, `C:\`) and the home folder itself;
* a system folder or a folder inside one: the list of nodestro-api's `output_folder.py` (`/bin`, `/etc`,
  `/usr`, `/var`, `/System`, `/Library`, `/private` ... with the exceptions `/var/home`, `/var/mnt`,
  `/var/tmp`, `/run/media`, `/private/tmp`, `/private/var/folders`) plus `/root`, and on Windows the folders
  that `SystemRoot`, `windir`, `ProgramFiles`, `ProgramFiles(x86)`, `ProgramW6432` and `ProgramData` name
  (`C:\Windows`, `C:\Program Files`, `C:\Program Files (x86)`, `C:\ProgramData` when they are not set);
* a `protected` folder (the product's data folder) or a folder inside one.

Folders are compared component by component (never by string prefix), NFC plus case-folded, and by identity, so
`/library`, a firmlink detour (`/System/Volumes/Data/...`) or a link to the data folder is refused too. A root
that *contains* the data folder (a home folder subfolder, say) is allowed; the data folder is then blocked below
it (`fs.blocked`).

Network paths raise `fs.network_path`: UNC shares (`\\server\share`), also administrative shares of this computer
(`\\localhost\C$`, `\\<computer name>\C$`, use `C:\` instead), `\\?\UNC\...`, mapped network drives
(`GetDriveTypeW` is `DRIVE_REMOTE`) and roots whose final path is a share. Windows device paths (`\\.\`, `\??\`,
`\\?\GLOBALROOT`, `\\?\Volume{...}`) raise `fs.invalid_path`; `\\?\C:\x` is `C:\x`.

**OneDrive** files on demand are cloud placeholders (reparse tag `IO_REPARSE_TAG_CLOUD*`, or the recall
attributes); reading one downloads it, so it is refused with `fs.cloud_file` unless the root is created with
`allow_cloud_files=True`. Then the file is opened again without `FILE_FLAG_OPEN_REPARSE_POINT` and must be the
same file (identity).

### Consistency with nodestro-runtime's access grants

N-06 (`tool_policy.folder_parts`) compares folder *names* and expects callers to resolve links first; `orkexa.fs`
is the file-system guard behind it. Both use NFC, whole components, case-folded comparison for denies and system
folders, the same system-folder environment variables, `\\?\C:\x` as `C:\x`, and refuse Windows device paths,
`:` in a component and 8.3 names in input. Differences, all failing closed: `SafeRoot` canonicalizes the root
(links resolved, 8.3 names expanded by the final path) where N-06 refuses a short name in a grant path; it refuses
every UNC root, where N-06 maps `\\localhost\C$` to drive `C:` for comparison; and it refuses relative names with a
trailing dot or space, where N-06 drops them as Windows does.

## Creating and renaming

With `writable=True`, `create_file(path, data)` creates a new file and `rename(source, target)` renames a file or
folder. Neither ever replaces an existing entry (`fs.exists`) or follows a link:

* POSIX: `openat(dirfd, name, O_CREAT | O_EXCL | O_NOFOLLOW)` in the checked folder; `renameat2(RENAME_NOREPLACE)`
  on Linux and `renameatx_np(RENAME_EXCL)` on macOS between the checked folders' descriptors. Without either (old
  kernels or file systems), a file is hard-linked to the new name, which fails if it exists, and unlinked.
* Windows: `CREATE_NEW` below the held folder handles, then the final path of the new handle;
  `MoveFileExW` without `MOVEFILE_REPLACE_EXISTING` and without `MOVEFILE_COPY_ALLOWED`.

Links, protected folders and files with several hard links are not renamed. A root is read only by default
(`fs.read_only`).

## Text extraction (PS-1)

`extract_text(path, budget=TextBudget(...))` reads a text file (`TEXT_SUFFIXES`: UTF-8, or UTF-16 with a byte
order mark; NUL bytes mean binary, `fs.unsupported_type`) or a PDF through the guard. PDFs go through
`orkexa.retrieval.pdf_pages_isolated`: a separate interpreter with a minimal environment that is killed at
`budget.timeout`, with `max_pages`, `max_chars` and, on Linux, `max_memory_mb`. A budget hit raises
`fs.too_complex` with `reason` `pages`, `chars`, `timeout`, `memory`, `limit` or `crash`; a PDF bomb (a page
that keeps the parser busy) ends within the time budget plus a few seconds of process start and kill. Office
formats are not part of Orkexa (Nodestro's runtime has its own DOCX and XLSX readers).

## Error codes

`FS_ERROR_CODES` maps every code to its meaning: `fs.invalid_path`, `fs.absolute_path`, `fs.parent_reference`,
`fs.link`, `fs.outside_root`, `fs.cross_device`, `fs.hard_link`, `fs.not_regular_file`, `fs.not_a_directory`,
`fs.is_a_directory`, `fs.not_found`, `fs.ambiguous_name`, `fs.blocked`, `fs.excluded`, `fs.too_large`,
`fs.too_many_entries`, `fs.protected_root`, `fs.network_path`, `fs.cloud_file`, `fs.root_changed`, `fs.revoked`,
`fs.read_only`, `fs.exists`, `fs.permission`, `fs.unreadable`, `fs.unsupported_type`, `fs.too_complex`.
`FsError` is a `PermissionError`; its message never contains an absolute path.

## The attack catalog

`orkexa.testing.pathguard.CATALOG` holds every N-10 case of the P0-2 attack catalog (`N10-01` to `N10-23`) and
every gap of the W0-6 spike (`W06-1` to `W06-5`), with the platforms where each is physically possible and the
codes that refuse it. Orkexa's tests (`tests/fs`) run all of them on Linux, macOS and Windows in CI. A product
runs the portable part against its own tools:

```python
from orkexa.testing import pathguard

layout = pathguard.build_layout(tmp_path)  # root/, outside/ with a canary, links where possible
assert pathguard.check_reader(my_read_tool, layout) == []
assert pathguard.swap_race(my_read_tool, layout, iterations=10_000) == []
```

## Remaining risks

* **The guard reads as the user.** It decides which path may be opened; it is no sandbox. Code running in the
  same process can bypass it.
* **Network file systems on Linux and macOS** (NFS, SMB mounts) are refused only when they are mounted *below*
  a root (`fs.cross_device`); a root *on* such a mount is treated like a local folder, and the server decides
  what a name means.
* **Windows renames** go through `MoveFileExW` with a path below held folder handles: the source entry itself is
  not pinned, so a concurrent swap inside the root can rename a different entry of the same folder (never one
  outside, never through a link).
* **Hard links opted in** (`allow_hard_links=True`) may expose a file that also lives elsewhere; that is the
  operator's choice.
* **Bidirectional and zero-width characters** are readable; showing names escaped (`Entry.display`) is the
  caller's job.
* An independent review (G-5) is due before the guard protects real user data (N-10 acceptance).
