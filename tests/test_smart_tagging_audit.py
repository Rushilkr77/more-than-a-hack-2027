from scripts.smart_tagging_audit import audit_file


def write_xml(tmp_path, content):
    path = tmp_path / "tags.xml"
    path.write_text(content, encoding="utf-8")
    return path


def test_audit_accepts_instances_and_ignores_the_display_colour_table(tmp_path):
    path = write_xml(tmp_path, """<?xml version="1.0"?>
<file>
  <ALL_INSTANCES>
    <instance>
      <ID>8</ID><start>12.0</start><end>16.0</end><code>BALL IN THE BOX</code>
      <label><group>Team</group><text>Barcelona</text></label>
      <label><group>Half</group><text>1st Half</text></label>
      <label><group>Direction of ball entry</group><text>Diagonal</text></label>
      <label><group>Max Players in the box</group><text>4</text></label>
    </instance>
  </ALL_INSTANCES>
  <ROWS><row><code>BALL IN THE BOX</code><R>1</R><G>2</G><B>3</B></row></ROWS>
</file>""")

    report = audit_file(path)

    assert report.instance_count == 1
    assert report.errors == []


def test_audit_reports_each_incomplete_or_unsafe_instance_with_its_id(tmp_path):
    path = write_xml(tmp_path, """<file><ALL_INSTANCES>
  <instance>
    <ID>17</ID><start>not-a-time</start><end>8.0</end><code>SET PIECES</code>
    <label><group>Team</group><text>N/A</text></label>
    <label><group>Half</group><text>3rd Half</text></label>
    <label><group>Type</group><text>Corner Kick</text></label>
    <label><group>Type</group><text>Throw-in</text></label>
  </instance>
  <instance><ID>18</ID><start>10</start><end>10</end><code>NOT A CODE</code></instance>
</ALL_INSTANCES></file>""")

    report = audit_file(path)
    issues = {(issue.instance_id, issue.rule) for issue in report.issues}

    assert ("17", "invalid_start") in issues
    assert ("17", "invalid_half") in issues
    assert ("17", "duplicate_label_group") in issues
    assert ("18", "non_positive_duration") in issues
    assert ("18", "unknown_code") in issues
    assert ("18", "missing_team") in issues
    assert ("18", "missing_half") in issues


def test_audit_marks_na_team_as_a_semantic_gap_not_a_malformed_label(tmp_path):
    path = write_xml(tmp_path, """<file><ALL_INSTANCES><instance>
      <ID>19</ID><start>10</start><end>15</end><code>SET PIECES</code>
      <label><group>Team</group><text>N/A</text></label>
      <label><group>Half</group><text>1st Half</text></label>
      <label><group>Type</group><text>Throw-in</text></label>
    </instance></ALL_INSTANCES></file>""")

    report = audit_file(path)

    assert report.errors == []
    assert ("19", "team_not_available") in {(issue.instance_id, issue.rule) for issue in report.issues}


def test_audit_accepts_other_team_names_and_validates_confidence(tmp_path):
    path = write_xml(tmp_path, """<file><ALL_INSTANCES>
      <instance><ID>20</ID><start>10</start><end>15</end><code>GOALS</code>
        <label><group>Team</group><text>Home FC</text></label>
        <label><group>Half</group><text>1st Half</text></label>
        <label><group>Confidence</group><text>0.90</text></label>
      </instance>
      <instance><ID>21</ID><start>20</start><end>25</end><code>GOALS</code>
        <label><group>Team</group><text>Away FC</text></label>
        <label><group>Half</group><text>2nd Half</text></label>
        <label><group>Confidence</group><text>1.2</text></label>
      </instance>
    </ALL_INSTANCES></file>""")

    report = audit_file(path)
    issues = {(issue.instance_id, issue.rule) for issue in report.issues}

    assert ("20", "unknown_label_value") not in issues
    assert ("20", "invalid_confidence") not in issues
    assert ("21", "invalid_confidence") in issues


def test_audit_turns_an_unparseable_xml_file_into_a_locatable_issue(tmp_path):
    path = write_xml(tmp_path, "<file><ALL_INSTANCES><instance></ALL_INSTANCES></file>")

    report = audit_file(path)

    assert report.instance_count == 0
    assert len(report.errors) == 1
    assert report.errors[0].rule == "xml_parse_error"
    assert report.errors[0].line is not None
